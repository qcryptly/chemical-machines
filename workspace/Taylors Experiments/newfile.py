import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

# 1. Device Agnostic Code
# This code automatically selects the best available hardware (GPU or CPU)
if torch.cuda.is_available():
    device = torch.device("cuda") # Use NVIDIA GPU
    print("CUDA is available. Training on GPU.")
elif torch.backends.mps.is_available():
    device = torch.device("mps") # Use Apple silicon GPU
    print("MPS is available. Training on Apple GPU.")
else:
    device = torch.device("cpu") # Use CPU
    print("No GPU found. Training on CPU.")

# 2. Define a simple model
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(10, 50)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(50, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

model = SimpleNN()
# Move the model to the target device
model.to(device)

# 3. Create dummy data and move to device
X_train = torch.randn(100, 10, device=device)
y_train = torch.randn(100, 1, device=device)

train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)

# 4. Define loss and optimizer
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 5. Training loop
num_epochs = 10
for epoch in range(num_epochs):
    for inputs, targets in train_loader:
        # Note: Inputs and targets are already on the correct device 
        # because the original tensors were created on the device.

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')