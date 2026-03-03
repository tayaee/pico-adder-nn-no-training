import torch
import torch.nn as nn

# 1. Define simple linear model
class PicoLinRegAdder(nn.Module):
    def __init__(self):
        super(PicoLinRegAdder, self).__init__()
        self.fc = nn.Linear(2, 1)

    def forward(self, x):
        return self.fc(x)

model = PicoLinRegAdder()

# 2. Inject weights and bias (Weight 1.0, Bias 0.0 for addition)
with torch.no_grad():
    model.fc.weight.copy_(torch.tensor([[1.0, 1.0]]))
    model.fc.bias.copy_(torch.tensor([0.0]))

print("Completed building linear regresssion model (weights/biases injected).")
print(f"W: {model.fc.weight.data}, B: {model.fc.bias.data}\n")

# 3. Test cases (e.g., 1-9 combinations)
test_cases = [[float(i), float(j)] for i in range(1, 10) for j in range(1, 10)]

# 4. Evaluation
model.eval()
print(f"{'Input':<8} | {'Prediction':<10} | {'Status'}")
print("-" * 32)

with torch.no_grad():
    for case in test_cases:
        x = torch.tensor([case])
        prediction = model(x).item()  # Changed 'out' to 'prediction'
        target = sum(case)
        status = "OK" if prediction == target else "FAIL"
        print(f"{int(case[0])}+{int(case[1])}={'?':<4} | {prediction:<10.1f} | {status}")
        