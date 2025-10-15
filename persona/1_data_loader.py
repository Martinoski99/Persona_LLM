"""
Load conversation data from YAML file

Each conversation has:
- user_message: "I'm feeling sad..."
- responses:
  - therapist: "I understand your feelings..."
  - friend: "That sucks, I'm here for you..."
  - non_compassionate: "Stop complaining..."
"""

import yaml
from pathlib import Path


def load_conversation_data(yaml_path):
    """Load YAML file with conversation examples"""
    print(f"Loading data from: {yaml_path}")
    
    if not Path(yaml_path).exists():
        print(f"ERROR: File not found: {yaml_path}")
        return None
    
    with open(yaml_path, 'r', encoding='utf-8') as f:
        conversations = yaml.safe_load(f)
    
    print(f"Loaded {len(conversations)} conversations")
    return conversations


def prepare_data_for_vector_generation(conversations):
    """Separate data into lists for vector generation"""
    prompts = []
    therapist_responses = []
    non_compassionate_responses = []
    
    for conv in conversations:
        prompts.append(conv['user_message'])
        therapist_responses.append(conv['responses']['therapist'])
        non_compassionate_responses.append(conv['responses']['non_compassionate'])
    
    print(f"Prepared {len(prompts)} conversations")
    return prompts, therapist_responses, non_compassionate_responses


def prepare_friend_data_for_vector_generation(conversations):
    """Separate data for friend responses"""
    prompts = []
    friend_responses = []
    non_compassionate_responses = []
    
    for conv in conversations:
        prompts.append(conv['user_message'])
        friend_responses.append(conv['responses']['friend'])
        non_compassionate_responses.append(conv['responses']['non_compassionate'])
    
    return prompts, friend_responses, non_compassionate_responses


def main():
    """Test the data loader"""
    conversations = load_conversation_data("conversation_data.yaml")
    
    if conversations is None:
        print("Could not load data")
        return
    
    # Show first example
    first = conversations[0]
    print(f"\nFirst conversation:")
    print(f"User: {first['user_message'][:50]}...")
    print(f"Therapist: {first['responses']['therapist'][:50]}...")
    
    # Prepare data
    prompts, therapist_responses, non_compassionate_responses = prepare_data_for_vector_generation(conversations)
    print(f"Ready for vector generation")


if __name__ == "__main__":
    main()