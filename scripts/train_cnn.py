import copy
import sys
import time
from pathlib import Path

# Add project root to Python path for imports
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from src.config import MODEL_PATH, PROJECT_ROOT
from src.models.cnn_definition import SimpleCNN

# --- Configuration ---
DATA_DIR = PROJECT_ROOT / "data"
MODEL_SAVE_PATH = MODEL_PATH
NUM_CLASSES = 13  # 12 pieces + 1 empty
BATCH_SIZE = 32  # Increase to 64 or 128 if using GPU for faster training
NUM_EPOCHS = 15
EARLY_STOPPING_PATIENCE = 5  # Stop if no improvement for 5 epochs
LEARNING_RATE = 0.001
INPUT_SIZE = 64  # Must match IMAGE_SIZE in data_generator.py
FORCE_CPU = False  # Set to True to force CPU usage even if GPU is available


# --- Data Loading and Transforms ---
def get_dataloaders(
    data_dir: Path,
    batch_size: int,
    input_size: int,
    use_gpu: bool = False,
) -> tuple[DataLoader, DataLoader, list[str]]:
    """Creates training and validation dataloaders.
    Raises:
        ValueError: If the number of classes in the training set does not match the expected number of classes.
        FileNotFoundError: If the required data directories do not exist.
    """
    # Check if data directories exist
    train_dir = data_dir / "train"
    valid_dir = data_dir / "validation"

    if not train_dir.exists() or not valid_dir.exists():
        error_msg = (
            f"Error: Required data directories not found.\n"
            f"  Expected: {train_dir}\n"
            f"  Expected: {valid_dir}\n\n"
            f"To generate training data, please follow these steps:\n"
            f"  1. Capture empty square backgrounds:\n"
            f"     python scripts/capture_empty_squares.py\n\n"
            f"  2. Generate synthetic training dataset:\n"
            f"     python src/data_handling/data_generator.py\n\n"
            f"Alternatively, manually organize your images in the following structure:\n"
            f"  {data_dir}/\n"
            f"    train/\n"
            f"      wP/, wN/, wB/, wR/, wQ/, wK/\n"
            f"      bP/, bN/, bB/, bR/, bQ/, bK/\n"
            f"      empty/\n"
            f"    validation/\n"
            f"      wP/, wN/, wB/, wR/, wQ/, wK/\n"
            f"      bP/, bN/, bB/, bR/, bQ/, bK/\n"
            f"      empty/\n"
        )
        raise FileNotFoundError(error_msg)

    data_transforms = {
        "train": transforms.Compose(
            [
                # Can add more augmentations here if needed
                # transforms.RandomRotation(5),
                # transforms.RandomHorizontalFlip(),
                transforms.Resize((input_size, input_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    [0.485, 0.456, 0.406],
                    [0.229, 0.224, 0.225],
                ),  # ImageNet stats
            ]
        ),
        "validation": transforms.Compose(
            [
                transforms.Resize((input_size, input_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        ),
    }

    print("Initializing Datasets and Dataloaders...")

    image_datasets = {
        x: datasets.ImageFolder(str(data_dir / x), data_transforms[x])
        for x in ["train", "validation"]
    }
    # Use 0 workers on Windows to avoid multiprocessing issues
    # On Linux/Mac, you can use 4-8 workers for faster data loading
    import platform

    num_workers = 0 if platform.system() == "Windows" else 4

    dataloaders = {
        x: DataLoader(
            image_datasets[x],
            batch_size=batch_size,
            shuffle=(x == "train"),
            num_workers=num_workers,
            pin_memory=use_gpu,  # Faster data transfer to GPU (only useful with GPU)
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


# --- Training Loop ---
def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    num_epochs: int = 15,
    early_stopping_patience: int = 5,
) -> nn.Module:
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    epochs_without_improvement = 0

    # Device selection with detailed information
    if FORCE_CPU:
        device = torch.device("cpu")
        print("⚠️  Forcing CPU usage (FORCE_CPU=True)")
    elif torch.cuda.is_available():
        device = torch.device("cuda:0")
        print(f"✅ GPU detected: {torch.cuda.get_device_name(0)}")
        print(
            f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB"
        )
        print(f"   CUDA Version: {torch.version.cuda}")
    else:
        device = torch.device("cpu")
        print("⚠️  No GPU detected. Training on CPU (this will be slower).")
        print("   To use GPU, ensure you have:")
        print("   - NVIDIA GPU with CUDA support")
        print(
            "   - PyTorch with CUDA installed (pip install torch --index-url https://download.pytorch.org/whl/cu118)"
        )

    print(f"🚀 Training on device: {device}\n")
    model.to(device)

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print("-" * 10)

        # Each epoch has a training and validation phase
        should_stop = False
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
            if phase == "validation":
                if epoch_acc > best_acc:
                    best_acc = epoch_acc.item()
                    best_model_wts = copy.deepcopy(model.state_dict())
                    epochs_without_improvement = 0
                    print(
                        f"New best validation accuracy: {best_acc:.4f}, saving model..."
                    )
                    torch.save(model.state_dict(), MODEL_SAVE_PATH)
                else:
                    epochs_without_improvement += 1
                    print(
                        f"No improvement for {epochs_without_improvement} epoch(s). Best: {best_acc:.4f}"
                    )

                # Early stopping check
                if epochs_without_improvement >= early_stopping_patience:
                    print(
                        f"\nEarly stopping triggered after {epoch + 1} epochs (no improvement for {early_stopping_patience} epochs)"
                    )
                    should_stop = True
                    break

        if should_stop:
            break

    time_elapsed = time.time() - since
    print(f"\nTraining complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
    print(f"Best val Acc: {best_acc:4f}")

    # Load best model weights
    model.load_state_dict(best_model_wts)
    return model


# --- Main Execution ---
if __name__ == "__main__":
    # Check GPU availability at startup
    if not FORCE_CPU and torch.cuda.is_available():
        print("=" * 60)
        print("GPU TRAINING ENABLED")
        print("=" * 60)
    else:
        print("=" * 60)
        print("CPU TRAINING MODE")
        print("=" * 60)
    print()

    # 1. Load Data
    # Check if we'll use GPU (before creating dataloaders for pin_memory optimization)
    will_use_gpu = not FORCE_CPU and torch.cuda.is_available()
    train_loader, valid_loader, class_names = get_dataloaders(
        DATA_DIR,
        BATCH_SIZE,
        INPUT_SIZE,
        use_gpu=will_use_gpu,
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
        early_stopping_patience=EARLY_STOPPING_PATIENCE,
    )

    print(f"\nFinished Training. Best model saved to {MODEL_SAVE_PATH}")
