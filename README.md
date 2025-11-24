# Mini GPT From Scratch

This repo provides a complete educational implementation of a GPT-style language model built from the ground up using PyTorch. This project demonstrates every component of modern LLMs, from tokenization to text generation.

## 📦 Project setup

1. Clone the repository: `git clone https://github.com/DangCongKhai/Mini_GPT_From_Scratch.git`
2. Dependencies: 
- Install uv from [here](https://docs.astral.sh/uv/installation/)
- Install the dependencies: In your terminal, run `uv sync`


## 🎯 What You'll Learn

- **Byte Pair Encoding (BPE)**: Build a custom tokenizer from scratch using the same algorithm as GPT-2/GPT-3
- **Self-Attention Mechanism**: Understand how transformers process sequential data
- **Positional Encoding**: Learn how models encode position information
- **GPT Architecture**: Implement the decoder-only transformer architecture with masked multi-head attention

## 🏗️ Project Structure

```
├── tokenizer.py       # BPE tokenizer implementation (V1 & V2)
├── gpt.py            # Complete GPT model architecture
├── main.ipynb        # End-to-end training notebook
├── dataset/          # Educational texts from Project Gutenberg
└── results/          # Training metrics and checkpoints
```

## 🚀 Key Features

- **Custom Tokenizer**: Two implementations of BPE with special token support
- **Modular Architecture**: 
  - Positional encoding layer
  - Masked multi-head self-attention
  - Feed-forward networks with GELU activation
  - Layer normalization and residual connections
- **Training Pipeline**: Complete workflow from data loading to text generation
- **Educational Focus**: Clear, well-commented code for learning


## 🎓 Model Configuration

Depending on your hardware, you can adjust the following parameters to train the model:

- Vocabulary Size: 2,000 tokens
- Context Window: 512 tokens
- Model Dimension: 256
- Attention Heads: 4
- Decoder Layers: 3
- Training Data: Educational philosophy texts

## 💡 Usage

1. **Train Tokenizer**: Build BPE vocabulary from your text corpus. Check 'tokenizer.py' for the implementation.
2. **Train Model**: Run the training loop with your dataset. Check 'gpt.py' for the implementation.
3. **Generate Text**: Use the trained model for text generation. Check 'main.ipynb' for the implementation.

## 📚 Materials


**Tokenizer**
- [Byte Pair Encoding Hugging Face](https://www.youtube.com/watch?v=HEikzVL-lZU)
- [Coding LLM Tokenizer From Scratch](https://www.youtube.com/watch?v=rsy5Ragmso8&list=PLPTV0NXA_ZSgsLAr8YCgCwhPIJNNtexWu&index=8)

**Positional Encoding**
- [Positional Encoding in Transformer Models](https://kazemnejad.com/blog/transformer_architecture_positional_encoding/)

**Model**
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [LLM from Scratch](https://github.com/rasbt/LLMs-from-scratch)