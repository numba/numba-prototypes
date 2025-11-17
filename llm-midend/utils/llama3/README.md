# Llama3 NumPy Implementation

This is a pure NumPy implementation of the Llama 3 model using the [stories15M model](https://huggingface.co/karpathy/tinyllamas/tree/main) trained by Andrej Karpathy, which is a smaller version of Llama2 designed for educational purposes.

## Requirements

```bash
pip install -r requirements.txt
```

The only dependency is NumPy (version 1.24.0 or higher).

## Setup

### 1. Download Pre-converted Files

Download the pre-converted model and tokenizer files from [likejazz/llama3.np](https://github.com/likejazz/llama3.np):

```bash
# Download pre-converted model weights
wget https://github.com/likejazz/llama3.np/raw/main/stories15M.model.npz

# Download pre-converted tokenizer
wget https://github.com/likejazz/llama3.np/raw/main/tokenizer.model.np
```

Note: While [hscspring/llama.np](https://github.com/hscspring/llama.np) provides conversion scripts (`convert_bin_llama_to_np.py` and `convert_tokenizer.py`), they may not work out of the box with the downloaded model.

## Usage

Run inference with a prompt:

```bash
python llama3.py "Your prompt here"
```

Example:
```bash
python llama3.py "I have a dream"
```

## Model Architecture

The implementation uses the following parameters for the stories15M model:
- Dimension: 288
- Hidden dimension: 768
- Number of layers: 6
- Number of attention heads: 6
- Number of KV heads: 6
- Vocabulary size: 32000
- Maximum sequence length: 256

## Implementation Details

This implementation:
- Uses pure NumPy for all operations
- Implements the Llama 3 architecture with rotary positional embeddings
- Uses greedy decoding (argmax) for text generation, ensuring deterministic outputs
- Implements key components:
  - RMSNorm for layer normalization
  - Rotary Positional Embeddings (RoPE)
  - Multi-head attention with KV caching
  - SwiGLU activation in feed-forward networks
- Supports the stories15M model format from Karpathy's tinyllamas

## References

This implementation is based on:
- [llama3.np](https://github.com/likejazz/llama3.np) by @likejazz - Main implementation reference
- [llama.np](https://github.com/hscspring/llama.np) by @hscspring - Additional implementation details
- [llama2.c](https://github.com/karpathy/llama2.c) by @karpathy - Original C implementation
- [modeling_llama.py](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py) from Hugging Face's Transformers

## License

MIT