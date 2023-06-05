import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Generate synthetic data
np.random.seed(0)
n_samples = 1000
n_channels = 32  # number of EEG channels
n_timepoints = 100  # number of timepoints per sample
X = np.random.normal(0, 1, (n_samples, n_channels, n_timepoints))
y = np.random.choice([0, 1], n_samples)

# Split data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Normalize data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
X_test = scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)

# Convert data to PyTorch tensors
X_train, y_train = torch.tensor(X_train).float(), torch.tensor(y_train).long()
X_test, y_test = torch.tensor(X_test).float(), torch.tensor(y_test).long()

# Create PyTorch dataloaders
train_data = TensorDataset(X_train, y_train)
test_data = TensorDataset(X_test, y_test)
batch_size = 32
train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
test_loader = DataLoader(test_data, shuffle=True, batch_size=batch_size)

# Define the CNN model
class EEGNet(nn.Module):
    def __init__(self):
        super(EEGNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=(n_channels, 1), stride=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=(1, n_timepoints), stride=1)
        self.fc1 = nn.Linear(32, 32)
        self.fc2 = nn.Linear(32, 2)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, (1, 4))
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Initialize the model, loss function, and optimizer
model = EEGNet().to('cpu')
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
n_epochs = 100
for epoch in range(n_epochs):
    for inputs, labels in train_loader:
        inputs, labels = inputs.to('cpu'), labels.to('cpu')

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

    # Print loss every 10 epochs
    if epoch % 10 == 0:
        print(f'Epoch {epoch}/{n_epochs}, Loss: {loss.item()}')

# Evaluate the model
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to('cpu'), labels.to('cpu')

        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Test Accuracy: {100 * correct / total}%')
