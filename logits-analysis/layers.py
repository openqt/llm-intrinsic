#! /usr/bin/env python3

import argparse
import time
import functools
from typing import Dict, Any, Callable

import torch
import pandas as pd
from transformers import LlamaForCausalLM, LlamaConfig, LlamaTokenizerFast as LlamaTokenizer
from transformers.modeling_outputs import CausalLMOutputWithPast


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
    parser.add_argument('--temperature', type=float, default=1.0,
                        help='Sampling temperature')
    parser.add_argument('--top-p', type=float, default=None,
                        help='Top-p sampling parameter')
    parser.add_argument('--top-k', type=int, default=None,
                        help='Top-k sampling parameter')
    parser.add_argument('--max-length', type=int, default=1000,
                        help='Maximum length of the generated text')
    return parser.parse_args()


def print_metrics(metrics: Dict[str, Any]) -> None:
    # Calculate generation statistics
    metrics['total_time'] = time.perf_counter() - metrics['start_time']
    metrics['generated_tokens'] = metrics['output_tokens'] - metrics['input_tokens']
    metrics['generation_speed'] = metrics['generated_tokens'] / metrics['generate_time']

    # Timing Summary DataFrame
    timing_data = [
        ['Model loading', metrics['load_model_time']],
        ['Generation', metrics['generate_time']],
    ]
    timing_df = pd.DataFrame(timing_data, columns=['Operation', 'Time (seconds)'])
    timing_df['Time (seconds)'] = timing_df['Time (seconds)'].round(2)
    
    print("\nTiming Summary:")
    print(timing_df.to_string(index=False))
    
    # Generation Statistics DataFrame
    stats_data = [
        ['Input length', f"{metrics['input_tokens']} tokens"],
        ['Output length', f"{metrics['output_tokens']} tokens"],
        ['Generated tokens', f"{metrics['generated_tokens']} tokens"],
        ['Generation speed', f"{metrics['generation_speed']:.2f} tk/s"]
    ]
    stats_df = pd.DataFrame(stats_data, columns=['Metric', 'Tokens'])
    
    print("\nGeneration Statistics:")
    print(stats_df.to_string(index=False))


@measure_time("load_model")
def load_model(model_path: str) -> LlamaForCausalLM:
    print("Loading model and tokenizer ...")
    config = LlamaConfig.from_pretrained(model_path)
    config.output_layer_logits = True

    tokenizer = LlamaTokenizer.from_pretrained(model_path, trust_remote_code=False)

    # Model configuration DataFrame
    config_data = [
        ['num_hidden_layers', config.num_hidden_layers],
        ['num_attention_heads', config.num_attention_heads],
        ['hidden_size', config.hidden_size],
        ['vocab_size', config.vocab_size],
        ['max_position_embeddings', config.max_position_embeddings],
        ['num_key_value_heads', config.num_key_value_heads]
    ]
    config_df = pd.DataFrame(config_data, columns=['Parameter', 'Value'])
    
    print("\nModel Configuration:")
    print(config_df.to_string(index=False))

    model = LlamaForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    return model, tokenizer


def get_layer_logits(model: LlamaForCausalLM, outputs: CausalLMOutputWithPast):
    layer_logits = []
    for layer in outputs.hidden_states:
        l = []
        for hidden_state in layer:
            l.append(model.lm_head(hidden_state[:, -1, :]))
        layer_logits.append(l)
    return layer_logits


def get_topk_layer_logits_with_tokenizer(current_step_logits, tokenizer, k: int = 5, layer_idx: int = None):
    """
    Get top-k token IDs and probabilities from layer logits with proper token decoding for a specific generation step.
    
    Args:
        layer_logits_steps: The layer logits steps to analyze.
        tokenizer: The tokenizer to use for decoding tokens
        k (int): Number of top tokens to return (default: 5)
        step_idx (int, optional): Specific generation step to analyze. If None, uses the last step.
        layer_idx (int, optional): Specific layer to analyze. If None, returns for all layers.
        
    Returns:
        dict: Dictionary containing top-k tokens and probabilities for each layer.
                Format: {layer_idx: {'token_ids': tensor, 'probs': tensor, 'tokens': list}}
    """
    results = {}
    
    # Determine which layers to process
    layers_to_process = [layer_idx] if layer_idx is not None else range(len(current_step_logits))
    
    for layer in layers_to_process:
        if layer >= len(current_step_logits):
            continue
            
        logits = current_step_logits[layer]  # Shape: [batch_size, 1, vocab_size]
        
        # Squeeze the middle dimension to get [batch_size, vocab_size]
        if logits.dim() == 3:
            logits = logits.squeeze(1)  # Remove the middle dimension
        
        # Get top-k tokens and probabilities
        probs = torch.softmax(logits, dim=-1)
        top_probs, top_indices = torch.topk(probs, k, dim=-1)
        
        # Decode tokens using the provided tokenizer
        tokens = []
        for idx in top_indices[0]:  # Take first batch
            try:
                token = tokenizer.decode([idx.item()])
                tokens.append(token)
            except:
                tokens.append(f"token_{idx.item()}")
        
        results[layer] = {
            'token_ids': top_indices,
            'probs': top_probs,
            'tokens': tokens
        }
    
    return results


def print_score(output_score, output_logit, tokenizer, k: int = 5):
    """
    Analyze the scores from the model.
    """
    # Apply top-k directly to output logits
    probs_score = torch.softmax(output_score, dim=-1)
    top_probs, top_indices = torch.topk(probs_score, k=k, dim=-1)
    
    probs_logit = torch.softmax(output_logit, dim=-1)
    top_probs_logit, top_indices_logit = torch.topk(probs_logit, k=k, dim=-1)
    
    # Decode tokens using tokenizer
    tokens = []
    for idx in top_indices:
        try:
            token = tokenizer.decode([idx.item()])
            tokens.append(repr(token))
        except:
            tokens.append(f"token_{idx.item()}")
    
    # Create output logits string in one line
    score_strings = []
    for i, (token_id, prob, token) in enumerate(zip(top_indices, top_probs, tokens)):
        score_strings.append(f"{token} ({prob:.4f})")
    
    logit_strings = []
    for i, (token_id, prob, token) in enumerate(zip(top_indices_logit, top_probs_logit, tokens)):
        logit_strings.append(f"{token} ({prob:.4f})")
    
    print("Output Logits:", " | ".join(logit_strings))
    print("Output Scores:", " | ".join(score_strings))
    print()
        

def print_layer_logits(layer_logits_step, tokenizer, k: int = 5):
    """
    Print the layer logits.
    """
    topk_results = get_topk_layer_logits_with_tokenizer(layer_logits_step, tokenizer, k=k)
    
    # Prepare data for DataFrame - each token in separate column
    layer_data = []
    for layer_idx, layer_data_dict in topk_results.items():
        token_ids = layer_data_dict['token_ids'][0]  # First batch
        probs_score = layer_data_dict['probs'][0]  # First batch
        tokens = layer_data_dict['tokens']
        
        # Create row data for this layer
        row_data = {'Layer': layer_idx}
        for i, (token_id, prob, token) in enumerate(zip(token_ids, probs_score, tokens)):
            row_data[f'Token {i+1}'] = f"{repr(token)} ({prob:.4f})"
        
        layer_data.append(row_data)
    
    if layer_data:
        step_df = pd.DataFrame(layer_data)
        print(step_df.to_string(index=False))
    else:
        print("No layer data available for this step.")


def analyze_layer_logits(model: LlamaForCausalLM, tokenizer: LlamaTokenizer, args: argparse.Namespace):
    """
    Analyze the top-k tokens from each layer's logits for all generation steps.
    """
    inputs = tokenizer(args.prompt, return_tensors="pt").to(model.device)
    k = args.top_k

    outputs = model.generate(
        **inputs,
        max_length=args.max_length,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=k,
        do_sample=True if args.temperature > 1e-6 else False,

        return_dict_in_generate=True,
        output_scores=True,
        output_logits=True,
        output_hidden_states=True,
        # output_attentions=True,
    )

    print(f"LAYER LOGITS ANALYSIS (Top-{k}) - All Generation Steps")
    print(f"{'='*80}")
    
    layer_logits_steps = get_layer_logits(model, outputs)
    if layer_logits_steps is None or len(layer_logits_steps) == 0:
        print("No layer logits available. Make sure to use output_layer_logits=True during generation.")
        return
    
    # Get output logits (final logits from the model)
    output_scores, output_logits = outputs.scores, outputs.logits
    assert len(layer_logits_steps) == len(output_scores) == len(output_logits)
    
    # Get the generated tokens to show in step titles
    for step_idx in range(len(layer_logits_steps)):
        # Get the generated token for this step
        generated_token = tokenizer.decode(outputs.sequences[0][step_idx+inputs.input_ids.shape[1]], 
                                           skip_special_tokens=True)
        
        # Display output logits for this step if available
        print("-" * 30, f"Step {step_idx} -> '{generated_token}'", "-" * 30)
        if output_scores is not None and step_idx < len(output_scores):
            print_score(output_scores[step_idx][0], output_logits[step_idx][0], tokenizer, k=k)
        print_layer_logits(layer_logits_steps[step_idx], tokenizer, k=k)

    metrics['input_tokens'] = inputs.input_ids.shape[-1]
    metrics['output_tokens'] = outputs.sequences.shape[-1]

    generated_tokens = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
    print("\nGenerated tokens:")
    print(">", generated_tokens)


@measure_time("total")
def main():
    global metrics
    args = parse_args()
    
    # Load tokenizer and model
    start_time = time.perf_counter()
    metrics['start_time'] = start_time

    model, tokenizer = load_model(args.model_path)
    metrics['total_loading_time'] = time.perf_counter() - start_time

    # Analyze layer logits
    analyze_layer_logits(model, tokenizer, args)
    metrics['generate_time'] = time.perf_counter() - start_time
    
    # Print all metrics
    print_metrics(metrics)


if __name__ == "__main__":
    main()
