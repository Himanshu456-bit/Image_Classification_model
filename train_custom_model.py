"""
Custom Image Classification Model Training
------------------------------------------
Trains a Convolutional Neural Network (CNN) across 10 distinct object classes:
['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']
Saves model weights ('simple_model.pth') and class labels ('classes.json').
"""

import json
import os
import torch
import torch.nn as nn
import torch.optim as optim

CLASSES = [
    "Airplane",
    "Automobile",
    "Bird",
    "Cat",
    "Deer",
    "Dog",
    "Frog",
    "Horse",
    "Ship",
    "Truck"
]

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=len(CLASSES)):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 32x32 -> 16x16

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 16x16 -> 8x8
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 8 * 8, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)

def train(save_dir=None):
    if save_dir is None:
        save_dir = os.path.dirname(os.path.abspath(__file__))

    print(f"--- Training CNN on {len(CLASSES)} Categories ---")
    for idx, name in enumerate(CLASSES):
        print(f"  Class {idx}: {name}")

    model = SimpleCNN(num_classes=len(CLASSES))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Generate synthetic training batch: 120 samples across 10 categories
    torch.manual_seed(42)
    inputs = torch.randn(120, 3, 32, 32)
    labels = torch.randint(0, len(CLASSES), (120,))

    model.train()
    for epoch in range(6):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        print(f"Epoch [{epoch + 1}/6] - Loss: {loss.item():.4f}")

    # Save model weights & class list
    model_path = os.path.join(save_dir, "simple_model.pth")
    classes_path = os.path.join(save_dir, "classes.json")

    torch.save(model.state_dict(), model_path)
    with open(classes_path, "w") as f:
        json.dump(CLASSES, f, indent=2)

    print(f"\nSaved 10-class model to: {model_path}")
    print(f"Saved class labels to: {classes_path}")

if __name__ == "__main__":
    train()
