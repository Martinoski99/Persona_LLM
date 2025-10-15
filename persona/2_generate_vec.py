"""
Generate compassion vectors using Gemma model

This loads conversation data, extracts hidden states, and calculates difference vectors.
The difference vectors tell us how to steer the model to be more/less compassionate.
"""

import argparse
import os
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

import importlib.util
spec = importlib.util.spec_from_file_location("data_loader", "1_data_loader.py")
data_loader = importlib.util.module_from_spec(spec)
spec.loader.exec_module(data_loader)


def get_hidden_states(model, tokenizer, prompts, responses):
    """Extract hidden states from the model for conversation data"""
    print(f"Extracting hidden states from {len(prompts)} examples...")
    
    max_layer = model.config.num_hidden_layers
    prompt_avg = [[] for _ in range(max_layer + 1)]
    response_avg = [[] for _ in range(max_layer + 1)]
    prompt_last = [[] for _ in range(max_layer + 1)]
    
    texts = [p + r for p, r in zip(prompts, responses)]
    
    for text, prompt in tqdm(zip(texts, prompts), total=len(texts)):
        inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False).to(model.device)
        prompt_len = len(tokenizer.encode(prompt, add_special_tokens=False))
        
        outputs = model(**inputs, output_hidden_states=True)
        
        for layer in range(max_layer + 1):
            prompt_avg[layer].append(outputs.hidden_states[layer][:, :prompt_len, :].mean(dim=1).detach().cpu())
            response_avg[layer].append(outputs.hidden_states[layer][:, prompt_len:, :].mean(dim=1).detach().cpu())
            prompt_last[layer].append(outputs.hidden_states[layer][:, prompt_len - 1, :].detach().cpu())
        
        del outputs
    
    for layer in range(max_layer + 1):
        prompt_avg[layer] = torch.cat(prompt_avg[layer], dim=0)
        prompt_last[layer] = torch.cat(prompt_last[layer], dim=0)
        response_avg[layer] = torch.cat(response_avg[layer], dim=0)
    
    print("Hidden states extracted")
    return prompt_avg, prompt_last, response_avg


def generate_compassion_vectors(model_name, data_path, save_dir):
    """Generate compassion vectors"""
    print(f"Starting compassion vector generation")
    print(f"Model: {model_name}")
    print(f"Data: {data_path}")
    print(f"Save to: {save_dir}")
    
    # Load model
    print(f"Loading model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        device_map="auto", 
        torch_dtype=torch.float16,
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    print(f"Model loaded ({model.config.num_hidden_layers} layers)")
    
    # Load data
    data = data_loader.load_conversation_data(data_path)
    prompts, pos_responses, neg_responses = data_loader.prepare_data_for_vector_generation(data)
    
    # Extract hidden states for therapist responses
    print("Processing therapist responses...")
    prompt_avg_pos, prompt_last_pos, response_avg_pos = get_hidden_states(model, tokenizer, prompts, pos_responses)
    
    # Extract hidden states for non-compassionate responses
    print("Processing non-compassionate responses...")
    prompt_avg_neg, prompt_last_neg, response_avg_neg = get_hidden_states(model, tokenizer, prompts, neg_responses)
    
    # Calculate difference vectors
    print("Computing difference vectors...")
    
    compassion_prompt_avg_diff = torch.stack([
        prompt_avg_pos[l].mean(0).float() - prompt_avg_neg[l].mean(0).float()
        for l in range(len(prompt_avg_pos))
    ], dim=0)
    
    compassion_response_avg_diff = torch.stack([
        response_avg_pos[l].mean(0).float() - response_avg_neg[l].mean(0).float()
        for l in range(len(response_avg_pos))
    ], dim=0)
    
    compassion_prompt_last_diff = torch.stack([
        prompt_last_pos[l].mean(0).float() - prompt_last_neg[l].mean(0).float()
        for l in range(len(prompt_last_pos))
    ], dim=0)
    
    # Do the same for friend responses
    print("Processing friend responses...")
    prompts_friend, pos_responses_friend, neg_responses_friend = data_loader.prepare_friend_data_for_vector_generation(data)
    _, _, response_avg_friend = get_hidden_states(model, tokenizer, prompts_friend, pos_responses_friend)
    
    compassion_response_friend_avg_diff = torch.stack([
        response_avg_friend[l].mean(0).float() - response_avg_neg[l].mean(0).float()
        for l in range(len(response_avg_friend))
    ], dim=0)
    
    # Save vectors
    os.makedirs(save_dir, exist_ok=True)
    print(f"Saving vectors to: {save_dir}")
    
    torch.save(compassion_prompt_avg_diff, f"{save_dir}/compassion_prompt_avg_diff.pt")
    torch.save(compassion_response_avg_diff, f"{save_dir}/compassion_response_therapist_avg_diff.pt")
    torch.save(compassion_prompt_last_diff, f"{save_dir}/compassion_prompt_last_diff.pt")
    torch.save(compassion_response_friend_avg_diff, f"{save_dir}/compassion_response_friend_avg_diff.pt")
    
    print("Compassion vector generation completed")
    print(f"Generated vectors for {model.config.num_hidden_layers} layers")
    print(f"Vector shape: [{model.config.num_hidden_layers}, {model.config.hidden_size}]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate compassion vectors")
    parser.add_argument("--model_name", type=str, required=True, help="Model name")
    parser.add_argument("--data_path", type=str, required=True, help="YAML data file")
    parser.add_argument("--save_dir", type=str, required=True, help="Where to save vectors")
    
    args = parser.parse_args()
    
    generate_compassion_vectors(
        model_name=args.model_name,
        data_path=args.data_path,
        save_dir=args.save_dir
    )