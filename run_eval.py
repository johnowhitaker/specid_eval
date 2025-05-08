#!/usr/bin/env python3
"""
Species identification model evaluator
Evaluates OpenAI and Gemini models on a species identification task
"""

import os
import json
import base64
import random
import argparse
import io
import time
from tqdm import tqdm
from datasets import load_dataset
import google.genai as genai
from google.genai import types
from openai import OpenAI

# Function tool definition
ANSWER_TOOL = {
    "function": {
        "name": "select_answer",
        "description": "Return the index (0‑4) of the correct species name.",
        "parameters": {
            "type": "object",
            "properties": {
                "answer": {"type": "string", "enum": ["0", "1", "2", "3", "4"]}
            },
            "required": ["answer"]
        }
    }
}

def ask_openai(img_b64, opts, model_name):
    """Query OpenAI model with image and options"""
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    
    prompt = (
        "Look at the image and choose the correct species.\n"
        + "\n".join(f"{i}. {o}" for i, o in enumerate(opts))
    )
    msgs = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}
                },
            ],
        }
    ]

    resp = client.chat.completions.create(
        model=model_name,
        messages=msgs,
        tools=[{"type": "function", **ANSWER_TOOL}],
        tool_choice="auto",
        timeout=30,
    )

    call = resp.choices[0].message.tool_calls[0]
    ans = json.loads(call.function.arguments)["answer"]
    return int(ans)

def ask_gemini(img_b64, opts, model_name):
    """Query Gemini model with image and options"""
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    tools = types.Tool(function_declarations=[ANSWER_TOOL["function"]])
    tool_config = types.ToolConfig(
        function_calling_config=types.FunctionCallingConfig(
            mode="ANY", allowed_function_names=["select_answer"] # Force function call
        )
    )
    config = types.GenerateContentConfig(tools=[tools], tool_config=tool_config)

    text = (
        "Choose the correct species for this photo and "
        "call select_answer with an integer 0‑4 only.\n"
        + "\n".join(f"{i}. {o}" for i, o in enumerate(opts))
    )

    image_part = {
        "mime_type": "image/jpeg",
        "data": img_b64
    }
    
    response = client.models.generate_content(
        model=model_name,
        contents=[
            {"role": "user", "parts": [
                {"text": text},
                {"inline_data": image_part}
            ]}
        ],
        config=config
    )
    
    if response.candidates[0].content.parts[0].function_call:
        function_call = response.candidates[0].content.parts[0].function_call
        ans = function_call.args["answer"]
        return int(ans)
    else:
        raise ValueError("No function call found in the response")

def evaluate_model(model_name, num_examples=None, seed=42, output_file=None):
    """
    Evaluate a model on the species identification task
    
    Args:
        model_name: Model identifier
        num_examples: Number of examples to evaluate (None = all)
        seed: Random seed for reproducibility
        output_file: File to save results (None = auto-generate)
    """
    random.seed(seed)
    
    # Load dataset
    ds = load_dataset("johnowhitaker/specid_eval_jw", split="train")
    
    # Select examples
    sample_size = len(ds) if num_examples is None else min(num_examples, len(ds))
    indices = random.sample(range(len(ds)), sample_size)
    
    # Set up results tracking
    correct = 0
    results = []
    
    print(f"Evaluating {model_name} on {sample_size} examples...")
    
    for idx in tqdm(indices, desc=f"Evaluating {model_name}"):
        example = ds[idx]
        img = example['image'].convert('RGB')
        options = example['options']
        correct_answer = example['answer']
        
        # Convert image to base64
        buffered = io.BytesIO()
        img.save(buffered, format="JPEG")
        img_b64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        
        try:
            # Choose appropriate model function
            if "gemini" in model_name.lower():
                predicted_answer = ask_gemini(img_b64, options, model_name)
            else:
                predicted_answer = ask_openai(img_b64, options, model_name)
            
            # Record result
            is_correct = (predicted_answer == correct_answer)
            if is_correct:
                correct += 1
                
            results.append({
                "observation_id": example["observation_id"],
                "options": options,
                "correct_answer": correct_answer,
                "predicted_answer": predicted_answer,
                "is_correct": is_correct
            })
            
        except Exception as e:
            print(f"Error on example {idx}: {e}")
            # Count as incorrect
            results.append({
                "observation_id": example["observation_id"],
                "options": options,
                "correct_answer": correct_answer,
                "predicted_answer": None,
                "is_correct": False,
                "error": str(e)
            })
    
    # Calculate accuracy
    accuracy = correct / sample_size
    
    # Print results
    print(f"\n===== EVALUATION RESULTS =====")
    print(f"Model: {model_name}")
    print(f"Samples: {sample_size}")
    print(f"Correct: {correct}")
    print(f"Accuracy: {accuracy:.2%}")
    
    # Prepare results
    evaluation_results = {
        "model": model_name,
        "accuracy": accuracy,
        "correct": correct,
        "total": sample_size,
        "results": results
    }
    
    # Save results to file
    if output_file is None:
        output_file = f"results_{model_name.replace('-', '_').replace('.', '_')}.json"
    
    with open(output_file, "w") as f:
        json.dump(evaluation_results, f, indent=2, default=str)
    
    print(f"Results saved to {output_file}")
    
    return evaluation_results

def main():
    parser = argparse.ArgumentParser(description="Evaluate models on species identification")
    parser.add_argument("model_name", help="Model to evaluate")
    parser.add_argument("--samples", type=int, default=None, 
                        help="Number of samples to evaluate (default: all)")
    parser.add_argument("--seed", type=int, default=42, 
                        help="Random seed (default: 42)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output file (default: auto-generated)")
    
    args = parser.parse_args()
    
    evaluate_model(
        model_name=args.model_name,
        num_examples=args.samples,
        seed=args.seed,
        output_file=args.output
    )

if __name__ == "__main__":
    main()
