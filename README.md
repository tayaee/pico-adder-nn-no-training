# Pico Adder: Model Hacking

This project shows how to create a functional AI model **without training**. Instead of backpropagation, we calculate and inject weights directly into the network.

## Quick Start
```bash
git clone https://github.com/tayaee/pico-adder-nn.git
uv run pico_transformer_adder.py
```

## The Core Idea: Bypassing Training
Traditional AI learns from data. **Model Hacking** assumes the logic is already known, so we bypass the "learning" phase and manually set the internal parameters.

1. **Analyze**: Define the logic (e.g., 1 + 3 = 4).
2. **Calculate**: Set weights to 1.0 and bias to 0.0.
3. **Inject**: Manually overwrite the model's memory.

## Key Implementation
Directly injecting values into PyTorch parameters:

```python
with torch.no_grad():
    model.fc.weight.copy_(torch.tensor([[1.0, 1.0]]))
    model.fc.bias.copy_(torch.tensor([0.0]))
```

## Why this matters?
Efficiency: Zero training time and data required.

## Example
Run main.py to see the "hacked" model achieve 100% accuracy instantly.
