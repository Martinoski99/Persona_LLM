"""
Activation Steering for Compassion Vectors

This applies steering vectors to the model during text generation.
The steering vectors push the model to be more or less compassionate.
"""

import torch
from contextlib import contextmanager


class ActivationSteerer:
    """
    Adds steering vector to model activations during generation.
    """
    
    def __init__(self, model, steering_vector, coeff=1.0, layer_idx=-1):
        self.model = model
        self.coeff = coeff
        self.layer_idx = layer_idx
        self._handle = None
        
        # Convert vector to tensor
        p = next(model.parameters())
        self.vector = torch.as_tensor(steering_vector, dtype=p.dtype, device=p.device)
    
    def _locate_layer(self):
        """Find the transformer layer to hook"""
        # For Llama models, layers are stored in model.model.layers
        return self.model.model.layers[self.layer_idx]
    
    def _hook_fn(self, module, ins, out):
        """Add steering vector to layer outputs"""
        steer = self.coeff * self.vector
        
        if torch.is_tensor(out):
            t2 = out.clone()
            t2[:, -1, :] += steer.to(out.device)
            return t2
        elif isinstance(out, (tuple, list)):
            if torch.is_tensor(out[0]):
                head = out[0].clone()
                head[:, -1, :] += steer.to(head.device)
                return (head, *out[1:])
        
        return out
    
    def __enter__(self):
        """Start steering"""
        layer = self._locate_layer()
        self._handle = layer.register_forward_hook(self._hook_fn)
        return self
    
    def __exit__(self, *exc):
        """Stop steering"""
        self.remove()
    
    def remove(self):
        """Remove the hook"""
        if self._handle:
            self._handle.remove()
            self._handle = None


def generate_with_steering(model, tokenizer, prompt, steering_vector, layer=20, coeff=2.0, max_tokens=100):
    """Generate text with steering applied"""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with ActivationSteerer(model, steering_vector, coeff=coeff, layer_idx=layer):
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9
            )
    
    prompt_len = inputs["input_ids"].shape[1]
    generated_text = tokenizer.decode(output[0][prompt_len:], skip_special_tokens=True)
    
    return generated_text


def main():
    """Test the steering and save results to JSON"""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch
    import os
    import json
    import time
    
    # Load model
    model_name = os.environ.get("MODEL_DIR", "/home/m/martinzb/links/scratch/huggingface/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659")
    print("USING MODEL_DIR:", model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        local_files_only=True, 
        device_map="auto", 
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load steering vector
    steering_vector = torch.load("persona_vectors/compassion_response_therapist_avg_diff.pt")[20]
    
    # Test prompts
    prompts = [
        "I'm feeling really sad today. What should I do?",
        "My friend recently lost their job and feels hopeless. What should I say to them?",
        "I failed an important exam and can't stop blaming myself. How can I deal with it?",
        "Someone I know is going through depression, but I don't know how to help. What should I do?"
    ]
    
    # Generate responses for each prompt
    results = []
    layers = [20]
    coefs = [2.0, -2.0]  # positive and negative steering
    
    for prompt in prompts:
        print(f"\nProcessing: {prompt[:50]}...")
        
        # Generate baseline (no steering)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=100, do_sample=True, temperature=0.7)
        prompt_len = inputs["input_ids"].shape[1]
        baseline = tokenizer.decode(output[0][prompt_len:], skip_special_tokens=True)
        
        # Generate steered responses
        steered_responses = []
        for layer in layers:
            for coef in coefs:
                steered = generate_with_steering(model, tokenizer, prompt, steering_vector, layer=layer, coeff=coef, max_tokens=100)
                steered_responses.append({
                    "layer": layer,
                    "coef": coef,
                    "answer": steered
                })
        
        results.append({
            "prompt": prompt,
            "baseline": baseline,
            "steered": steered_responses
        })
    
    # Save results to JSON
    os.makedirs("results", exist_ok=True)
    timestamp = int(time.time())
    output_file = f"results/compassion_steering_results_{timestamp}.json"
    
    data = {
        "metadata": {
            "model": model_name,
            "vector_file": "persona_vectors/compassion_response_therapist_avg_diff.pt",
            "layers": layers,
            "coefs": coefs,
            "temperature": 0.7,
            "top_p": 0.9,
            "max_tokens": 100,
            "steering_type": "response",
            "timestamp": timestamp
        },
        "data": results
    }
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print(f"\nResults saved to: {output_file}")
    print(f"Generated {len(results)} prompts with {len(steered_responses)} steered responses each")


if __name__ == "__main__":
    main()