import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import os
import time

data_transforms = {
    "train": transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    "test": transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

def train_model(model, criterion, optimizer, dataloaders, dataset_sizes, num_epochs=3):
    start_time = time.time()
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        for phase in ["train", "test"]:
            if phase == "train":
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

    total_time = time.time() - start_time
    print(f"Training complete in {total_time // 60:.0f}m {total_time % 60:.0f}s")
    return model

if __name__ == "__main__":
    # Load full dataset from small_dataset_tiles
    data_dir = r"P:\petes_code\github\LunarPi\Training"  # Updated path
    image_datasets = {
        "train": datasets.ImageFolder(os.path.join(data_dir, "train"), data_transforms["train"]),
        "test": datasets.ImageFolder(os.path.join(data_dir, "test"), data_transforms["test"])
    }
    dataloaders = {
        "train": DataLoader(image_datasets["train"], batch_size=32, shuffle=True, num_workers=8, pin_memory=True),
        "test": DataLoader(image_datasets["test"], batch_size=32, shuffle=False, num_workers=8, pin_memory=True)
    }
    dataset_sizes = {x: len(image_datasets[x]) for x in ["train", "test"]}
    class_names = image_datasets["train"].classes
    print(f"Classes: {class_names}, Train: {dataset_sizes['train']}, Test: {dataset_sizes['test']}")

    # Load model
    model = models.resnet18(weights="IMAGENET1K_V1")
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 8)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # Train and save
    model = train_model(model, criterion, optimizer, dataloaders, dataset_sizes, num_epochs=10)
    torch.save(model.state_dict(), "rock_classifier_full.pth")
    print("Model saved as rock_classifier_full.pth")