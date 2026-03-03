import torch
import torch.nn as nn
import torch.nn.functional as F

class PicoTransformerAdder(nn.Module):
    def __init__(self, d_model=4):
        super().__init__()
        self.d_model = d_model
        # 1. Input Embedding (convert numbers into vectors)
        self.embedding = nn.Linear(1, d_model)
        
        # 2. Simple Attention Layer (core structure from the article)
        self.query = nn.Linear(d_model, d_model)
        self.key   = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        
        # 3. Feed Forward Network (includes one hidden layer as requested)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 8), # Hidden Layer
            nn.ReLU(),
            nn.Linear(8, 1)        # Output
        )

    def forward(self, x):
        # x shape: (batch, 2) -> treat each number as a separate token
        x = x.unsqueeze(-1) # (batch, 2, 1)
        x = self.embedding(x) # (batch, 2, d_model)
        
        # Attention computation (simplified)
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / (self.d_model ** 0.5)
        attn_probs = F.softmax(attn_weights, dim=-1)
        out = torch.matmul(attn_probs, v)
        
        # Take the mean to merge into a single vector, then pass through FFN
        out = out.mean(dim=1)
        return self.ffn(out)

# Create model
model = PicoTransformerAdder(d_model=4)

# 2. Manually inject weights (Hand-set weights magic)
# Forcefully inject addition logic.
with torch.no_grad():

    # ------------------
    # 1. Embedding
    # ------------------
    model.embedding.weight.zero_()
    model.embedding.bias.zero_()
    model.embedding.weight[0, 0] = 1.0

    # ------------------
    # 2. Attention (pure averaging)
    # ------------------
    model.query.weight.zero_()
    model.query.bias.zero_()

    model.key.weight.zero_()
    model.key.bias.zero_()

    model.value.weight.zero_()
    model.value.bias.zero_()
    for i in range(model.d_model):
        model.value.weight[i, i] = 1.0

    # ------------------
    # 3. FFN
    # ------------------

    B = 1000.0

    # Linear(d_model → 8)
    model.ffn[0].weight.zero_()
    model.ffn[0].bias.zero_()

    # Use only the first hidden neuron
    # h = 2 * first_dim + B
    model.ffn[0].weight[0, 0] = 2.0
    model.ffn[0].bias[0] = B

    # Linear(8 → 1)
    model.ffn[2].weight.zero_()
    model.ffn[2].bias.zero_()

    # output = h - B
    model.ffn[2].weight[0, 0] = 1.0
    model.ffn[2].bias[0] = -B

print("Finished injecting weights into the tiny Qwen3-style transformer.\n")

# 3. Run test
test_cases = [[1.0, 2.0], [5.0, 5.0], [10.0, 20.0]]
model.eval()

for case in test_cases:
    inp = torch.tensor([case], dtype=torch.float32)
    pred = model(inp).item()
    print(f"Input: {case[0]} + {case[1]} | Prediction: {pred:.2f}")