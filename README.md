# Pico Adder: Model Hacking

This project shows how to create a functional AI model **without training**. Instead of backpropagation, we calculate and inject weights directly into the network.

## Quick Start
```bash
% git clone https://github.com/tayaee/pico-adder-nn-no-training.git
% cd pico-adder-nn-no-training
% uv run pico_transformer_adder.py
Finished injecting weights into the tiny Qwen3-style transformer.

Input: 1.0 + 2.0 | Prediction: 3.00
Input: 5.0 + 5.0 | Prediction: 10.00
Input: 10.0 + 20.0 | Prediction: 30.00
```

## Why this matters?
Efficiency: Zero training time and data required.
