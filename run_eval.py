#!/usr/bin/env python3
"""
Species identification model evaluator
Evaluates models on a species identification task

Supports:
- OpenAI (via OpenAI Python SDK)
- Gemini (native google.genai) or via OpenRouter (OpenAI-compatible)
- Grok (xAI SDK)
- Cohere (via HuggingFace InferenceClient)
"""

import os
import json
import base64
import random
import argparse
import io
import time
from tqdm import tqdm
import re
import requests
from datasets import load_dataset

# Function tool definition for OpenAI/Gemini
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
        tool_choice="required",
        timeout=300,
    )

    call = resp.choices[0].message.tool_calls[0]
    ans = json.loads(call.function.arguments)["answer"]
    return int(ans)

def ask_openrouter(img_b64, opts, model_name, reasoning_enabled=False):
    """Query a model via OpenRouter's OpenAI-compatible API.

    Expects OPENROUTER_API_KEY in environment. Uses OpenAI SDK with custom base_url.
    Falls back to parsing numeric answer from text if the model doesn't call the tool.
    """
    if "OPENROUTER_API_KEY" not in os.environ:
        raise RuntimeError("OPENROUTER_API_KEY not set for OpenRouter usage")

    prompt = (
        "Look at the image and choose the correct species. "
        "Respond by calling the function only.\n"
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

    payload = {
        "model": model_name,
        "messages": msgs,
        "tools": [{"type": "function", **ANSWER_TOOL}],
        "tool_choice": "auto",
    }
    if reasoning_enabled:
        payload["reasoning"] = {"enabled": True}

    resp = requests.post(
        url="https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}",
            "Content-Type": "application/json",
        },
        data=json.dumps(payload),
        timeout=300,
    )
    resp.raise_for_status()
    data = resp.json()

    choice = data.get("choices", [{}])[0]
    message = choice.get("message", {})

    # Tool call path
    tool_calls = message.get("tool_calls") or []
    if tool_calls:
        call = tool_calls[0]
        fn = call.get("function", {})
        args = fn.get("arguments", "{}")
        ans = json.loads(args).get("answer")
        return int(ans)

    # Fallback: parse text content
    text = (message.get("content") or "").strip()
    m = re.search(r"\b([0-4])\b", text)
    if m:
        return int(m.group(1))

    raise ValueError("Model did not return a tool call or parseable answer")

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

def ask_grok(img_b64, opts, model_name):
    """Query Grok model with image and options"""
    client = Client(api_key=os.environ["XAI_API_KEY"])
    
    prompt = (
        "Look at the image and choose the correct species.\n"
        + "\n".join(f"{i}. {o}" for i, o in enumerate(opts))
    )
    
    chat = client.chat.create(
        model=model_name,
        tools=GROK_TOOL_DEFINITIONS,
        tool_choice="required",
    )
    
    chat.append(user(prompt, image(image_url=f"data:image/jpeg;base64,{img_b64}", detail="high")))
    response = chat.sample()
    
    if response.tool_calls:
        tool_call = response.tool_calls[0]
        ans = json.loads(tool_call.function.arguments)["answer"]
        return int(ans)
    else:
        raise ValueError("No tool call found in the response")

def ask_cohere(img_b64, opts, model_name):
    """Query Cohere model with image and options"""
    client = InferenceClient(
        provider="cohere",
        api_key=os.environ["HF_TOKEN"],
    )
    
    prompt = (
        "Look at the image and choose the correct species.\n"
        + "\n".join(f"{i}. {o}" for i, o in enumerate(opts))
    )
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            completion = client.chat.completions.create(
                model=model_name,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{img_b64}"
                                }
                            }
                        ]
                    }
                ],
                tools=[{"type": "function", **ANSWER_TOOL}],
                tool_choice="auto",
            )
            
            # Extract answer from function call
            if completion.choices[0].message.tool_calls:
                call = completion.choices[0].message.tool_calls[0]
                ans = json.loads(call.function.arguments)["answer"]
                return int(ans)
            else:
                raise ValueError("No function call found in the response")
            
        except Exception as e:
            if "429" in str(e) and attempt < max_retries - 1:
                # Rate limited, wait and retry
                wait_time = 2 ** attempt  # Exponential backoff
                time.sleep(wait_time)
                continue
            else:
                raise e

def evaluate_model(model_name, num_examples=None, seed=42, output_file=None, reasoning=False):
    """
    Evaluate a model on the species identification task
    
    Args:
        model_name: Model identifier
        num_examples: Number of examples to evaluate (None = all)
        seed: Random seed for reproducibility
        output_file: File to save results (None = auto-generate)
    """
    # Import required API based on model name
    global OpenAI, genai, types, Client, user, image, tool, InferenceClient

    has_openrouter = bool(os.getenv("OPENROUTER_API_KEY"))
    use_openrouter_for_gemini = ("gemini" in model_name.lower() and has_openrouter)
    use_openrouter_default = (
        has_openrouter and not ("grok" in model_name.lower() or "cohere" in model_name.lower())
    )

    if (not use_openrouter_for_gemini) and "gemini" in model_name.lower():
        import google.genai as genai
        from google.genai import types
    elif "grok" in model_name.lower():
        from xai_sdk import Client
        from xai_sdk.chat import user, image, tool
        
        # Define Grok tool when needed
        global GROK_TOOL_DEFINITIONS
        GROK_TOOL_DEFINITIONS = [
            tool(
                name="select_answer",
                description="Return the index (0‑4) of the correct species name.",
                parameters={
                    "type": "object",
                    "properties": {
                        "answer": {"type": "string", "enum": ["0", "1", "2", "3", "4"]}
                    },
                    "required": ["answer"]
                }
            )
        ]
    elif "cohere" in model_name.lower():
        from huggingface_hub import InferenceClient
    else:  # OpenAI models
        from openai import OpenAI
    
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
            # Route through OpenRouter when available (covers OpenAI and Gemini)
            if use_openrouter_for_gemini or use_openrouter_default:
                # Enable reasoning automatically for models known to support it (e.g., gpt-5.2)
                auto_reasoning = ("gpt-5.2" in model_name)
                predicted_answer = ask_openrouter(
                    img_b64, options, model_name, reasoning_enabled=(reasoning or auto_reasoning)
                )
            elif "gemini" in model_name.lower():
                predicted_answer = ask_gemini(img_b64, options, model_name)
            elif "grok" in model_name.lower():
                predicted_answer = ask_grok(img_b64, options, model_name)
            elif "cohere" in model_name.lower():
                predicted_answer = ask_cohere(img_b64, options, model_name)
                # Add small delay for Cohere to avoid rate limiting
                time.sleep(0.5)
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
        # Clean model name for file path
        clean_name = model_name.replace('/', '_').replace('-', '_').replace('.', '_')
        output_file = f"results_{clean_name}.json"
    
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
    parser.add_argument("--reasoning", action="store_true", default=False,
                        help="Enable OpenRouter reasoning (for models like openai/gpt-5.2)")
    
    args = parser.parse_args()
    
    evaluate_model(
        model_name=args.model_name,
        num_examples=args.samples,
        seed=args.seed,
        output_file=args.output,
        reasoning=args.reasoning,
    )

if __name__ == "__main__":
    main()
