import copy
import time
from pathlib import Path

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# --- Configuration ---
DATA_DIR = Path("./data")
MODEL_SAVE_PATH = Path("./chess_piece_cnn.pth")
NUM_CLASSES = 13  # 12 pieces + 1 empty
BATCH_SIZE = 32
NUM_EPOCHS = 25  # Start with a reasonable number, can be adjusted
LEARNING_RATE = 0.001
INPUT_SIZE = 64  # Must match IMAGE_SIZE in data_generator.py


# --- Data Loading and Transforms ---
def get_dataloaders(
    data_dir: Path,
    batch_size: int,
    input_size: int,
) -> tuple[DataLoader, DataLoader, list[str]]:
    """Creates training and validation dataloaders.
    Raises:
        ValueError: If the number of classes in the training set does not match the expected number of classes.
    """
    data_transforms = {
        "train": transforms.Compose([
            # Can add more augmentations here if needed
            # transforms.RandomRotation(5),
            # transforms.RandomHorizontalFlip(),
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225],
            ),  # ImageNet stats
        ]),
        "validation": transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]),
    }

    print("Initializing Datasets and Dataloaders...")

    image_datasets = {
        x: datasets.ImageFolder(str(data_dir / x), data_transforms[x])
        for x in ["train", "validation"]
    }
    dataloaders = {
        x: DataLoader(
            image_datasets[x],
            batch_size=batch_size,
            shuffle=(x == "train"),
            num_workers=4,
        )
        for x in ["train", "validation"]
    }
    class_names = image_datasets["train"].classes

    if len(class_names) != NUM_CLASSES:
        msg = (
            f"Expected {NUM_CLASSES} classes based on config, but found {len(class_names)} "
            f"folders in {data_dir / 'train'}: {class_names}"
        )
        raise ValueError(
            msg,
        )

    print(f"Found classes: {class_names}")
    print(f"Training samples: {len(image_datasets['train'])}")
    print(f"Validation samples: {len(image_datasets['validation'])}")

    return dataloaders["train"], dataloaders["validation"], class_names


# --- Model Definition (Simple CNN) ---
class SimpleCNN(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(2, 2)
        # Calculate flattened size: 64 channels * (input_size / 2^3) * (input_size / 2^3)
        flattened_size = 64 * (INPUT_SIZE // 8) * (INPUT_SIZE // 8)
        self.fc1 = nn.Linear(flattened_size, 512)
        self.relu4 = nn.ReLU()
        self.dropout = nn.Dropout(0.5)  # Add dropout for regularization
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.pool3(self.relu3(self.conv3(x)))
        x = torch.flatten(x, 1)  # Flatten all dimensions except batch
        x = self.relu4(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)


# --- Training Loop ---
def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    num_epochs: int = 25,
) -> nn.Module:
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")
    model.to(device)

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print("-" * 10)

        # Each epoch has a training and validation phase
        for phase in ["train", "validation"]:
            if phase == "train":
                model.train()  # Set model to training mode
                dataloader = train_loader
            else:
                model.eval()  # Set model to evaluate mode
                dataloader = valid_loader

            running_loss = 0.0
            running_corrects = torch.tensor(0, device=device)
            total_samples = 0

            # Iterate over data.
            for i, (inputs, labels) in enumerate(dataloader):
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward
                # Track history only if in train
                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # Backward + optimize only if in training phase
                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                # Statistics
                batch_loss = loss.item() * inputs.size(0)
                batch_corrects = torch.sum(preds == labels.data)
                running_loss += batch_loss
                running_corrects += batch_corrects
                total_samples += inputs.size(0)

                if i % 10 == 0 and phase == "train":  # Print progress every 10 batches
                    print(
                        f"  Batch {i}/{len(dataloader)} Loss: {batch_loss / inputs.size(0):.4f} Acc: {batch_corrects / inputs.size(0):.4f}",
                    )

            epoch_loss = running_loss / total_samples
            epoch_acc = running_corrects.float() / total_samples

            print(f"{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

            # Deep copy the model if best validation accuracy so far
            if phase == "validation" and epoch_acc > best_acc:
                best_acc = epoch_acc.item()
                best_model_wts = copy.deepcopy(model.state_dict())
                print(f"New best validation accuracy: {best_acc:.4f}, saving model...")
                torch.save(model.state_dict(), MODEL_SAVE_PATH)

    time_elapsed = time.time() - since
    print(f"\nTraining complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
    print(f"Best val Acc: {best_acc:4f}")

    # Load best model weights
    model.load_state_dict(best_model_wts)
    return model


# --- Main Execution ---
if __name__ == "__main__":
    # 1. Load Data
    train_loader, valid_loader, class_names = get_dataloaders(
        DATA_DIR,
        BATCH_SIZE,
        INPUT_SIZE,
    )

    # 2. Initialize Model
    # model = models.resnet18(pretrained=True) # Example using pretrained
    # num_ftrs = model.fc.in_features
    # model.fc = nn.Linear(num_ftrs, NUM_CLASSES)
    model = SimpleCNN(num_classes=NUM_CLASSES)
    print("\nModel Architecture:")
    print(model)

    # 3. Define Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    # Observe that all parameters are being optimized
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    # optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # 4. Train the model
    trained_model = train_model(
        model,
        train_loader,
        valid_loader,
        criterion,
        optimizer,
        num_epochs=NUM_EPOCHS,
    )

    print(f"\nFinished Training. Best model saved to {MODEL_SAVE_PATH}")
