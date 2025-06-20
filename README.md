# llm-intrinsic

## logits-analysis

The `logits-analysis` module provides tools for analyzing the internal logits of transformer models during text generation. This analysis helps researchers and developers understand how different layers of the model contribute to the generation process and token predictions.

### Features

- **Layer-wise Logits Analysis**: Extract and analyze logits from each transformer layer during text generation
- **Top-k Token Analysis**: View the top-k most likely tokens predicted by each layer at each generation step
- **Performance Metrics**: Comprehensive timing and performance measurements for model loading, tokenization, generation, and decoding
- **Custom Llama Implementation**: Modified Llama model implementation that supports outputting layer logits during generation

### Components

- `layers.py`: Main script for running inference and analyzing layer logits
- `llama/`: Custom Llama model implementation with layer logits output capability
  - `modeling_llama.py`: Modified Llama model that captures layer logits
  - `configuration_llama.py`: Model configuration with logits output options
  - `tokenization_llama_fast.py`: Fast tokenizer implementation

### Usage

```bash
python logits-analysis/layers.py --model-path /path/to/llama/model --prompt "Your input prompt here"
```

### Key Parameters

- `--model-path`: Path to the Llama model directory
- `--prompt`: Input text prompt for generation
- `--temperature`: Sampling temperature (default: 0.7)
- `--top-p`: Top-p sampling parameter (default: 0.95)
- `--top-k`: Top-k sampling parameter (default: 5)
- `--max-length`: Maximum generation length (default: 1000)

### Output

The tool provides:

1. Generated text response
2. Layer-by-layer analysis of top-k predicted tokens for each generation step
3. Detailed performance metrics including timing breakdowns
4. Generation statistics (tokens per second, input/output lengths)

This analysis is particularly useful for:

- Understanding model behavior at different layers
- Debugging generation issues
- Research on transformer model interpretability
- Analyzing how different layers contribute to token predictions
