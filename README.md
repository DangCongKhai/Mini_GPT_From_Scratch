# Mini GPT From Scratch

This repo provides a complete educational implementation of a GPT-style language model built from the ground up using PyTorch. This project demonstrates every component of modern LLMs, from tokenization to text generation.

## Project Setup

1. Clone the repository: `git clone https://github.com/DangCongKhai/Mini_GPT_From_Scratch.git`
2. Dependencies: 
- Install uv from [here](https://docs.astral.sh/uv/installation/)
- Install the dependencies: In your terminal, run `uv sync`


## Project Structure

```
├── tokenizer.py       # BPE tokenizer implementation (V1 & V2)
├── gpt.py            # Complete GPT model architecture
├── main.ipynb        # End-to-end training notebook
├── dataset/          # Educational texts from Project Gutenberg
└── results/          # Training metrics and checkpoints
```

## Key Features

- **Custom Tokenizer**: Two implementations of BPE with special token support
- **Modular Architecture**: 
  - Positional encoding layer
  - Masked multi-head self-attention
  - Feed-forward networks with GELU activation
  - Layer normalization and residual connections
- **Training Pipeline**: Complete workflow from data loading to text generation



## Model Configuration

Depending on your hardware, you can adjust the following parameters to train the model:

- Vocabulary Size: 2,000 tokens
- Context Window: 512 tokens
- Model Dimension: 256
- Attention Heads: 4
- Decoder Layers: 3
- Training Data: Educational philosophy texts

## Model Performance

The model was trained for 8 epochs on educational philosophy texts:

| Metric | Training | Validation |
|--------|----------|------------|
| **Final Loss** | 1.043 | 0.628 |
| **Final Perplexity** | 2.84 | 1.88 |
| **Best Epoch** | 8 | 8 |

### Training Progress

- **Initial Loss**: 2.84 → **Final Loss**: 1.04 (63% improvement)
- **Initial Perplexity**: 21.89 → **Final Perplexity**: 2.84 (87% improvement)

## Usage

1. **Train Tokenizer**: Build BPE vocabulary from your text corpus. Check 'tokenizer.py' for the implementation.
2. **Train Model**: Run the training loop with your dataset. Check 'gpt.py' for the implementation.
3. **Generate Text**: Use the trained model for text generation. Check 'main.ipynb' for the implementation.

## Materials


**Tokenizer**
- [Byte Pair Encoding Hugging Face](https://www.youtube.com/watch?v=HEikzVL-lZU)
- [Coding LLM Tokenizer From Scratch](https://www.youtube.com/watch?v=rsy5Ragmso8&list=PLPTV0NXA_ZSgsLAr8YCgCwhPIJNNtexWu&index=8)

**Positional Encoding**
- [Positional Encoding in Transformer Models](https://kazemnejad.com/blog/transformer_architecture_positional_encoding/)

**Model**
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [LLM from Scratch](https://github.com/rasbt/LLMs-from-scratch)