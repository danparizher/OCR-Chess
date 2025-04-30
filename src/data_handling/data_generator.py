from __future__ import annotations

import random
from pathlib import Path

from PIL import Image, ImageEnhance

# --- Configuration ---
PIECE_DIR = Path("./python-chess-pieces")
# We will need to capture/generate these later
EMPTY_SQUARE_DIR = Path("./empty_squares")
OUTPUT_DIR = Path("./data")
TRAIN_DIR = OUTPUT_DIR / "train"
VALID_DIR = OUTPUT_DIR / "validation"
IMAGE_SIZE = (64, 64)  # Target size for each square image
AUGMENTATION_FACTOR = 10  # Number of augmented images per piece/background combo
VALIDATION_SPLIT = 0.2  # Percentage of data to use for validation

# Piece types including an 'empty' class
PIECE_TYPES = [
    "wP",
    "wN",
    "wB",
    "wR",
    "wQ",
    "wK",
    "bP",
    "bN",
    "bB",
    "bR",
    "bQ",
    "bK",
    "empty",
]

# --- Helper Functions ---


def _load_single_image(img_path: Path) -> tuple[str | None, Image.Image | None]:
    """Loads a single image, returning None if loading fails."""
    try:
        # Use stem for key (e.g., 'wP' from 'wP.png')
        img = Image.open(img_path).convert("RGBA")
    except Exception as e:
        print(f"Error loading image {img_path}: {e}")
        return None, None
    else:
        return img_path.stem, img


def load_images_from_dir(directory: Path) -> dict[str, Image.Image]:
    """Loads PNG images from a directory into a dictionary."""
    images: dict[str, Image.Image] = {}
    if not directory.is_dir():
        print(f"Warning: Directory not found: {directory}")
        return images
    for img_path in directory.glob("*.png"):
        stem, img = _load_single_image(img_path)
        if stem and img:
            images[stem] = img
    return images


def _resize_and_position_piece(
    piece_img: Image.Image,
    target_size: tuple[int, int],
) -> tuple[Image.Image, tuple[int, int]]:
    """Resizes the piece to fit and calculates its centered position."""
    # Resize piece to fit within the square (e.g., 80% of size)
    piece_max_dim = int(min(target_size) * 0.8)
    piece_resized = piece_img.copy()
    piece_resized.thumbnail(
        (piece_max_dim, piece_max_dim),
        Image.Resampling.LANCZOS,
    )

    # Calculate position to center the piece
    paste_x = (target_size[0] - piece_resized.width) // 2
    paste_y = (target_size[1] - piece_resized.height) // 2
    return piece_resized, (paste_x, paste_y)


def apply_random_augmentation(image: Image.Image) -> Image.Image:
    """Applies simple random augmentations to an image."""
    # Brightness
    brightness_enhancer = ImageEnhance.Brightness(image)
    augmented_image = brightness_enhancer.enhance(random.uniform(0.8, 1.2))

    # Contrast
    contrast_enhancer = ImageEnhance.Contrast(augmented_image)
    augmented_image = contrast_enhancer.enhance(random.uniform(0.8, 1.2))

    # Tiny rotation (e.g., +/- 2 degrees) - might be less useful for chess
    # angle = random.uniform(-2, 2)
    # image = image.rotate(angle, expand=True, fillcolor=(0,0,0,0)) # Ensure transparent bg

    # Slight scaling (e.g., 95% to 105%)
    scale = random.uniform(0.95, 1.05)
    new_size = (int(image.width * scale), int(image.height * scale))
    # Use LANCZOS for high-quality downscaling/upscaling
    return augmented_image.resize(new_size, Image.Resampling.LANCZOS)


def create_synthetic_image(
    piece_img: Image.Image | None,
    background_img: Image.Image,
    target_size: tuple[int, int],
) -> Image.Image:
    """Overlays a piece onto a background square, resizes, and applies augmentation."""
    bg_copy = (
        background_img.copy()
        .convert("RGBA")
        .resize(target_size, Image.Resampling.LANCZOS)
    )

    if piece_img:
        piece_resized, paste_pos = _resize_and_position_piece(piece_img, target_size)

        # Augment the piece before pasting
        # piece_augmented = apply_random_augmentation(piece_resized) # Augment piece only?

        # Paste piece using its alpha channel as mask
        bg_copy.paste(piece_resized, paste_pos, piece_resized)

    # Augment the final composite image
    final_image = apply_random_augmentation(bg_copy)

    return final_image.convert("RGB")  # Convert to RGB for saving as JPG/PNG


# --- Main Generation Logic ---


def generate_dataset() -> None:
    """Generates the synthetic dataset."""
    print("Loading piece templates...")
    piece_templates = load_images_from_dir(PIECE_DIR)
    if not piece_templates:
        print(f"Error: No piece templates found in {PIECE_DIR}")
        return

    print("Loading empty square backgrounds...")
    # !!! We need images of empty squares first !!!
    # For now, let's create placeholder solid color backgrounds
    # empty_squares = {
    #     "light_empty": Image.new("RGB", IMAGE_SIZE, (240, 217, 181)), # Light square color
    #     "dark_empty": Image.new("RGB", IMAGE_SIZE, (181, 136, 99)),   # Dark square color
    # }
    # TODO: Replace placeholders with actual loaded empty square images
    empty_squares = load_images_from_dir(EMPTY_SQUARE_DIR)
    if not empty_squares:
        print(f"Error: No empty square images found in {EMPTY_SQUARE_DIR}.")
        print("Please run capture_empty_squares.py first.")
        return  # Stop generation if no backgrounds are found

    print("Creating output directories...")
    for split_dir in [TRAIN_DIR, VALID_DIR]:
        for piece_type in PIECE_TYPES:
            (split_dir / piece_type).mkdir(parents=True, exist_ok=True)

    print("Generating synthetic images...")
    image_counter = 0
    for piece_name, piece_img in piece_templates.items():
        print(f"  Generating images for {piece_name}...")
        for bg_name, bg_img in empty_squares.items():
            for i in range(AUGMENTATION_FACTOR):
                synthetic_img = create_synthetic_image(piece_img, bg_img, IMAGE_SIZE)

                # Decide train/validation split
                output_base = (
                    VALID_DIR if random.random() < VALIDATION_SPLIT else TRAIN_DIR
                )
                output_path = (
                    output_base / piece_name / f"{piece_name}_on_{bg_name}_{i:03d}.png"
                )
                synthetic_img.save(output_path)
                image_counter += 1

    # Generate 'empty' square images using the backgrounds
    print("  Generating images for empty squares...")
    for bg_name, bg_img in empty_squares.items():
        for i in range(
            AUGMENTATION_FACTOR * len(piece_templates) // 2,
        ):  # Generate roughly same num as pieces
            # Create an "empty" image just from the background with augmentation
            synthetic_img = create_synthetic_image(None, bg_img, IMAGE_SIZE)

            output_base = VALID_DIR if random.random() < VALIDATION_SPLIT else TRAIN_DIR
            output_path = output_base / "empty" / f"empty_{bg_name}_{i:03d}.png"
            synthetic_img.save(output_path)
            image_counter += 1

    print(f"Generated {image_counter} total synthetic images in {OUTPUT_DIR}")


if __name__ == "__main__":
    generate_dataset()
