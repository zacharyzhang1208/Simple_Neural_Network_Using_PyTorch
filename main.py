import torch
import torch.nn as nn
import torch.optim as optim

# Training Data: XOR Data
train_X = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)
train_y = torch.tensor([[0], [1], [1], [0]], dtype=torch.float32)


# Define the neural network model
class XORModel(nn.Module):
    def __init__(self):
        super(XORModel, self).__init__()
        self.linear1 = nn.Linear(2, 3)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(3, 1)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x


# Initialize model, loss function and optimizer
model = XORModel()
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.005)

# Training
for epoch in range(10000):
    # Forward pass
    outputs = model(train_X)
    loss = criterion(outputs, train_y)

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 200 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item()}')

    if loss.item() < 0.001:
        print(f"Training completed! Iteration {epoch}")
        break

# Predict
print("-----")
X = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)
with torch.no_grad():
    result = model(X)
    predicted = (result > 0.5).float()

print("data:")
print(X)
print("result:")
print(predicted)