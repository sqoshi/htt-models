import copy
import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms


class ASLDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []

        for person in os.listdir(root_dir):
            person_dir = os.path.join(root_dir, person)
            if not os.path.isdir(person_dir):
                continue
            for label in os.listdir(person_dir):
                label_dir = os.path.join(person_dir, label)
                if not os.path.isdir(label_dir):
                    continue
                for img_name in os.listdir(label_dir):
                    if img_name.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
                        img_path = os.path.join(label_dir, img_name)
                        self.image_paths.append(img_path)
                        self.labels.append(int(label))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]

        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label


# Data augmentations and transformations
train_transforms = transforms.Compose(
    [
        transforms.RandomRotation(15),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225],
        ),
    ]
)

val_transforms = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)


def create_dataloaders(root_dir, batch_size=32, valid_size=0.2, test_size=0.1):
    dataset = ASLDataset(root_dir=root_dir, transform=train_transforms)

    # Split dataset into train, val, and test indices
    indices = list(range(len(dataset)))
    train_indices, test_indices = train_test_split(
        indices, test_size=test_size, stratify=dataset.labels, random_state=42
    )
    train_indices, val_indices = train_test_split(
        train_indices,
        test_size=valid_size / (1 - test_size),
        stratify=[dataset.labels[i] for i in train_indices],
        random_state=42,
    )

    # Create subsets
    train_subset = torch.utils.data.Subset(dataset, train_indices)
    val_subset = torch.utils.data.Subset(dataset, val_indices)
    test_subset = torch.utils.data.Subset(dataset, test_indices)

    # Update transforms
    train_subset.dataset.transform = train_transforms
    val_subset.dataset.transform = val_transforms
    test_subset.dataset.transform = val_transforms

    # Create DataLoaders with num_workers=0 to avoid multiprocessing overhead on CPU
    train_loader = DataLoader(
        train_subset, batch_size=batch_size, shuffle=True, num_workers=1
    )
    val_loader = DataLoader(
        val_subset, batch_size=batch_size, shuffle=False, num_workers=1
    )
    test_loader = DataLoader(
        test_subset, batch_size=batch_size, shuffle=False, num_workers=1
    )

    return train_loader, val_loader, test_loader


# Using ResNet18 Pretrained Model for ASL Classification
def initialize_resnet_model(num_classes=26):
    model = models.resnet18(pretrained=True)  # Load pretrained ResNet18
    num_ftrs = (
        model.fc.in_features
    )  # Get the number of input features to the fully connected layer
    model.fc = nn.Linear(
        num_ftrs, num_classes
    )  # Modify the final layer to classify 26 classes
    return model


# Train Model
def train_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    scheduler,
    num_epochs=25,
    device="cuda",
):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print("-" * 10)

        # Each epoch has a training and validation phase
        for phase in ["train", "val"]:
            if phase == "train":
                model.train()  # Set model to training mode
                dataloader = train_loader
            else:
                model.eval()  # Set model to evaluate mode
                dataloader = val_loader

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data
            for inputs, labels in dataloader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward pass
                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # Backward + optimize only in train phase
                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                # Statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == "train":
                scheduler.step()

            epoch_loss = running_loss / (len(dataloader.dataset))
            epoch_acc = running_corrects.double() / (len(dataloader.dataset))

            print(f"{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

            # Deep copy the model if it performs better on validation set
            if phase == "val" and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == "val" and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print(f"Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
    print(f"Best Validation Acc: {best_acc:.4f}")

    # Load best model weights
    model.load_state_dict(best_model_wts)
    return model


# Evaluate Model
def evaluate_model(model, test_loader, criterion, device="cuda"):
    model.eval()  # Set model to evaluation mode
    running_loss = 0.0
    running_corrects = 0

    # Do not calculate gradients for evaluation
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            # Statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

    # Calculate final loss and accuracy
    test_loss = running_loss / len(test_loader.dataset)
    test_acc = running_corrects.double() / len(test_loader.dataset)

    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")

    return test_loss, test_acc


# Main function to run the entire pipeline
def main():
    root_dir = "/home/piotr/Documents/htt/images224224"

    batch_size = 32
    num_epochs = 25
    learning_rate = 0.001
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create DataLoaders
    train_loader, val_loader, test_loader = create_dataloaders(
        root_dir, batch_size=batch_size, valid_size=0.2, test_size=0.1
    )

    # Initialize the ResNet18 model
    model = initialize_resnet_model(num_classes=26)
    model = model.to(device)

    # Define Loss, Optimizer, and Scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    # Train the model
    trained_model = train_model(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        scheduler,
        num_epochs=num_epochs,
        device=device,
    )

    # Save the trained model
    torch.save(trained_model.state_dict(), "Xresnet_asl_cnn_model.pth")
    print("ResNet18 ASL model saved as resnet_asl_cnn_model.pth")

    # Evaluate the model on the test set
    print("\nEvaluating ResNet18 on the test set...")
    evaluate_model(trained_model, test_loader, criterion, device=device)


def draw():
    import torch
    from torchviz import make_dot

    # Define a simple input tensor
    x = torch.randn(
        1, 3, 224, 224
    )  # Example input (batch size 1, 3 channels, 224x224 image)

    # Initialize the modified ResNet-18 model
    model = initialize_resnet_model(num_classes=26)

    # Forward pass through the model
    y = model(x)

    # Create the computation graph, focusing only on major layers
    # The params argument lists the parameters, which we can extract from the model,
    # while 'show_attrs=False' hides additional attributes that clutter the graph
    dot = make_dot(
        y, params=dict(model.named_parameters()), show_attrs=False, show_saved=False
    )

    # Output simplified graph
    dot.format = "png"
    dot.render("simplified_resnet18_major_layers")


if __name__ == "__main__":
    draw()
