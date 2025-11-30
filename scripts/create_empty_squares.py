"""Create placeholder empty square images for data generation."""

from pathlib import Path

from PIL import Image

# Configuration matching data_generator.py
EMPTY_SQUARE_DIR = Path("./data/empty_squares")
IMAGE_SIZE = (64, 64)  # Must match IMAGE_SIZE in data_generator.py

# Standard chess board colors
LIGHT_SQUARE_COLOR = (240, 217, 181)  # Light square color
DARK_SQUARE_COLOR = (181, 136, 99)  # Dark square color

NUM_LIGHT_SQUARES = 10
NUM_DARK_SQUARES = 10


def create_empty_squares() -> None:
    """Create placeholder empty square images."""
    print(f"Creating empty square directory: {EMPTY_SQUARE_DIR}")
    EMPTY_SQUARE_DIR.mkdir(parents=True, exist_ok=True)

    print("Generating light empty squares...")
    for i in range(NUM_LIGHT_SQUARES):
        img = Image.new("RGB", IMAGE_SIZE, LIGHT_SQUARE_COLOR)
        filename = f"empty_light_{i:02d}.png"
        img.save(EMPTY_SQUARE_DIR / filename)
        print(f"  Created: {filename}")

    print("Generating dark empty squares...")
    for i in range(NUM_DARK_SQUARES):
        img = Image.new("RGB", IMAGE_SIZE, DARK_SQUARE_COLOR)
        filename = f"empty_dark_{i:02d}.png"
        img.save(EMPTY_SQUARE_DIR / filename)
        print(f"  Created: {filename}")

    print(
        f"\nSuccessfully created {NUM_LIGHT_SQUARES + NUM_DARK_SQUARES} empty square images in {EMPTY_SQUARE_DIR}"
    )


if __name__ == "__main__":
    create_empty_squares()
