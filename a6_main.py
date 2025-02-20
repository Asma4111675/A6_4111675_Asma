import torch
import torch.optim as optim
from a6_model_[YOUR-FIRST-NAME] import TinyImageNetCNN
from a6_data import train_loader, test_loader, device


model = TinyImageNetCNN().to(device)
model = torch.nn.DataParallel(model) 


criterion = torch.nn.NLLLoss()


optimizer = optim.Adam(model.parameters(), lr=0.0001)

num_epochs = 50  

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f" Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")

model.eval()
correct = 0
total = 0

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f": {accuracy:.2f}%")

torch.save(model.state_dict(), "model_state_dict.pth")

import json

config = {
    "model_name": "TinyImageNetCNN",
    "input_size": (3, 64, 64),
    "num_classes": 200,
    "layers": [
        "Conv2d(3 -> 64, 11x11, stride=4, padding=2)", "ReLU",
        "MaxPool2d(3x3, stride=2)",
        "Conv2d(64 -> 192, 5x5, padding=2)", "ReLU",
        "MaxPool2d(3x3, stride=2)",
        "Conv2d(192 -> 384, 3x3, padding=1)", "ReLU",
        "Conv2d(384 -> 256, 3x3, padding=1)", "ReLU",
        "Conv2d(256 -> 256, 3x3, padding=1)", "ReLU",
        "MaxPool2d(3x3, stride=2)",
        "AdaptiveAvgPool2d(6x6)",
        "Flatten",
        "Dropout(0.5)",
        "Linear(256*6*6 -> 4096)", "ReLU",
        "Dropout(0.5)",
        "Linear(4096 -> 4096)", "ReLU",
        "Linear(4096 -> 200)", "LogSoftmax"
    ]
}

with open("config.json", "w") as f:
    json.dump(config, f, indent=4)

print("state_dict و config.json!")
