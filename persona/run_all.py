"""
Run the complete compassion steering pipeline

This runs all steps in sequence:
1. Generate steering vectors
2. Test steering with sample prompts
"""

import subprocess
import sys
import os


def run_command(command, description):
    """Run a command and show progress"""
    print(f"\n{'='*60}")
    print(f"STEP: {description}")
    print(f"{'='*60}")
    print(f"Running: {command}")
    
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    
    if result.returncode == 0:
        print("SUCCESS!")
        if result.stdout:
            print(result.stdout)
    else:
        print("ERROR!")
        print(result.stderr)
        return False
    
    return True


def main():
    """Run the complete pipeline"""
    print("COMPASSION STEERING PIPELINE")
    print("="*60)
    print("This will:")
    print("1. Generate steering vectors from your data")
    print("2. Test steering with sample prompts")
    print("="*60)
    
    # Check if files exist
    if not os.path.exists("conversation_data.yaml"):
        print("ERROR: conversation_data.yaml not found!")
        return
    
    if not os.path.exists("2_generate_vec.py"):
        print("ERROR: 2_generate_vec.py not found!")
        return
    
    if not os.path.exists("activation_steer.py"):
        print("ERROR: activation_steer.py not found!")
        return
    
    # Step 1: Generate vectors
    if not run_command(
        "python 2_generate_vec.py --model_name google/gemma-2-8b-it --data_path conversation_data.yaml --save_dir persona_vectors/",
        "Generate steering vectors"
    ):
        print("Failed to generate vectors. Stopping.")
        return
    
    # Check if vectors were created
    if not os.path.exists("persona_vectors/compassion_response_therapist_avg_diff.pt"):
        print("ERROR: Steering vectors not created!")
        return
    
    # Step 2: Test steering
    if not run_command(
        "python activation_steer.py",
        "Test steering"
    ):
        print("Failed to test steering.")
        return
    
    print(f"\n{'='*60}")
    print("PIPELINE COMPLETE!")
    print("="*60)
    print("You can now:")
    print("1. Check the persona_vectors/ folder for your steering vectors")
    print("2. Modify activation_steer.py to test different prompts")
    print("3. Experiment with different steering coefficients")
    print("="*60)


if __name__ == "__main__":
    main()

