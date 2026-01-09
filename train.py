# train.py (FINAL UPDATES)
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm


from dataset import RAFDBDataset
from model import ResEmoteNetCBAM 


# Config

BATCH_SIZE = 16
EPOCHS = 30
LR = 1e-4
NUM_CLASSES = 7

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


DATA_ROOT = "basic/Image/aligned" 
ANNOTATION_FILE = "basic/EmoLabel/list_patition_label.txt"

# Transforms

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Dataset & DataLoader

train_dataset = RAFDBDataset(
    root_dir=DATA_ROOT,
    annotation_file=ANNOTATION_FILE,
    transform=train_transform,
    train=True
)

test_dataset = RAFDBDataset(
    root_dir=DATA_ROOT,
    annotation_file=ANNOTATION_FILE,
    transform=test_transform,
    train=False
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=4,
    pin_memory=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=4,
    pin_memory=True
)

print(f"Train samples: {len(train_dataset)}")
print(f"Test samples: {len(test_dataset)}")

# Model, Loss, Optimizer

model = ResEmoteNetCBAM(num_classes=NUM_CLASSES).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# Initialize best accuracy and best model path for saving
best_accuracy = 0.0
MODEL_SAVE_PATH = "models/best_model.pth"


# Evaluation Function (Test/Validation Accuracy)
def evaluate(model, data_loader, device):
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    return accuracy

# Training Loop (Finalized)
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0.0
    correct_train = 0
    total_train = 0

    print(f"--- Epoch {epoch+1}/{EPOCHS} ---")
    for images, labels in tqdm(train_loader, desc=f"Training"):
        images = images.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        
        # Calculate Training Accuracy (Added)
        _, predicted_train = torch.max(outputs.data, 1)
        total_train += labels.size(0)
        correct_train += (predicted_train == labels).sum().item()


    # Calculate and Print Training Metrics
    avg_loss = total_loss / len(train_loader)
    train_accuracy = 100 * correct_train / total_train
    
    # Calculate and Print Test/Validation Accuracy
    test_accuracy = evaluate(model, test_loader, DEVICE)

    print(f"Metrics | Loss: {avg_loss:.4f} | Train Acc: {train_accuracy:.2f}% | Test Acc (Validation Score): {test_accuracy:.2f}%")

    # Save Best Model based on Test/Validation Accuracy
    if test_accuracy > best_accuracy:
        best_accuracy = test_accuracy
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
        print(f"⭐ Model saved! New best validation score: {best_accuracy:.2f}%")

# Final Output
print("==============================")
print("✅ Training complete.")
print(f"Best Test Accuracy achieved (Validation Score): {best_accuracy:.2f}%")
print(f"Final model saved to: {MODEL_SAVE_PATH}")