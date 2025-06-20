#! /usr/bin/env python3

import argparse
import time
import functools
from typing import Dict, Any, Callable

import torch
from llama import LlamaForCausalLM, LlamaConfig, LlamaTokenizerFast as LlamaTokenizer


# Global metrics dictionary
metrics: Dict[str, float] = {}


def measure_time(func_name: str = None) -> Callable:
    """Decorator to measure the execution time of a function.
    
    Args:
        func_name: Optional name for the metric. If None, uses the function name.
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            global metrics
            name = func_name or func.__name__
            start = time.perf_counter()
            result = func(*args, **kwargs)
            end = time.perf_counter()
            metrics[f"{name}_time"] = end - start
            return result
        return wrapper
    return decorator


def parse_args():
    parser = argparse.ArgumentParser(description='Run inference with a transformer model')
    parser.add_argument('--model-path', type=str, required=True,
                        help='Path to the model directory')
    parser.add_argument('--prompt', type=str, 
                        default="Write a short story about a robot learning to paint.",
                        help='Input prompt for generation')
    parser.add_argument('--temperature', type=float, default=0.7,
                        help='Sampling temperature')
    parser.add_argument('--top-p', type=float, default=0.95,
                        help='Top-p sampling parameter')
    parser.add_argument('--top-k', type=int, default=5,
                        help='Top-k sampling parameter')
    parser.add_argument('--max-length', type=int, default=1000,
                        help='Maximum length of the generated text')
    return parser.parse_args()


def print_metrics(metrics: Dict[str, Any]) -> None:
    # Calculate generation statistics
    metrics['total_time'] = time.perf_counter() - metrics['start_time']
    metrics['generated_tokens'] = metrics['output_tokens'] - metrics['input_tokens']
    metrics['generation_speed'] = metrics['generated_tokens'] / metrics['generate_time']

    print("\nTiming Summary:")
    print(f"Total model loading time: {metrics['total_loading_time']:.2f} seconds")
    print(f"├─ Tokenizer loading: {metrics['load_tokenizer_time']:.2f} seconds")
    print(f"└─ Model loading: {metrics['load_model_time']:.2f} seconds")
    
    print("\nProcessing Times:")
    print(f"Input tokenization: {metrics['tokenize_input_time']:.2f} seconds")
    print(f"Generation: {metrics['generate_time']:.2f} seconds")
    print(f"Decoding: {metrics['decode_time']:.2f} seconds")
    print(f"Total running time: {metrics['total_time']:.2f} seconds")
    
    print("\nGeneration Statistics:")
    print(f"Input length: {metrics['input_tokens']} tokens")
    print(f"Output length: {metrics['output_tokens']} tokens")
    print(f"Generated tokens: {metrics['generated_tokens']} tokens")
    print(f"Generation speed: {metrics['generation_speed']:.2f} tokens/second")


@measure_time("load_tokenizer")
def load_tokenizer(model_path: str) -> LlamaTokenizer:
    print("Loading tokenizer...")
    return LlamaTokenizer.from_pretrained(model_path, trust_remote_code=True)


@measure_time("load_model")
def load_model(model_path: str) -> LlamaForCausalLM:
    print("Loading model...")
    config = LlamaConfig.from_pretrained(model_path)
    config.output_layer_logits = True

    # print(f">>> embedding_size: {config.embedding_size}")
    print(f">>> num_hidden_layers: {config.num_hidden_layers}")
    print(f">>> num_attention_heads: {config.num_attention_heads}")
    print(f">>> hidden_size: {config.hidden_size}")
    print(f">>> vocab_size: {config.vocab_size}")
    print(f">>> max_position_embeddings: {config.max_position_embeddings}")
    print(f">>> num_key_value_heads: {config.num_key_value_heads}")

    return LlamaForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )


@measure_time("tokenize_input")
def tokenize_input(tokenizer: LlamaTokenizer, prompt: str, device: torch.device):
    return tokenizer(prompt, return_tensors="pt").to(device)


@measure_time("generate")
def generate_text(model: LlamaForCausalLM, inputs: Dict[str, torch.Tensor], args) -> torch.Tensor:
    result = model.generate(
        **inputs,
        max_length=args.max_length,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        do_sample=True if args.temperature > 1e-6 else False,

        output_layer_logits=True,
    )
    return result

@measure_time("decode")
def decode_output(tokenizer: LlamaTokenizer, outputs: torch.Tensor) -> str:
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    print("\nGenerated response:")
    print(response)


def analyze_layer_logits(model: LlamaForCausalLM, tokenizer: LlamaTokenizer, k: int = 5):
    """
    Analyze the top-k tokens from each layer's logits for all generation steps.
    """
    print(f"\n{'='*80}")
    print(f"LAYER LOGITS ANALYSIS (Top-{k}) - All Generation Steps")
    print(f"{'='*80}")
    
    layer_logits_steps = model.get_layer_logits()
    if layer_logits_steps is None or len(layer_logits_steps) == 0:
        print("No layer logits available. Make sure to use output_layer_logits=True during generation.")
        return
    
    for step_idx in range(len(layer_logits_steps)):
        print(f"\nGeneration Step {step_idx}:")
        print("-" * 50)
        topk_results = model.get_topk_layer_logits_with_tokenizer(tokenizer, k=k, step_idx=step_idx)
        for layer_idx, layer_data in topk_results.items():
            results = []
            token_ids = layer_data['token_ids'][0]  # First batch
            probs = layer_data['probs'][0]  # First batch
            tokens = layer_data['tokens']
            for i, (token_id, prob, token) in enumerate(zip(token_ids, probs, tokens)):
                results.append((token_id.item(), token))
            print(f"  Layer {layer_idx}: {results}")


@measure_time("total")
def main():
    global metrics
    args = parse_args()
    
    # Load tokenizer and model
    start_time = time.perf_counter()
    metrics['start_time'] = start_time

    tokenizer = load_tokenizer(args.model_path)
    model = load_model(args.model_path)
    metrics['total_loading_time'] = time.perf_counter() - start_time

    # Process the input and generate response
    inputs = tokenize_input(tokenizer, args.prompt, model.device)
    outputs = generate_text(model, inputs, args)
    decode_output(tokenizer, outputs)
    
    # Analyze layer logits
    analyze_layer_logits(model, tokenizer, k=5)
    
    # Print all metrics
    metrics['input_tokens'] = inputs.input_ids.shape[1]
    metrics['output_tokens'] = outputs.shape[1]
    print_metrics(metrics)


if __name__ == "__main__":
    main()
