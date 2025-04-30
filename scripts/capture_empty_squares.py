from pathlib import Path

from PIL import Image

from src.utils.ocr_utils import load_cnn_model
from src.utils.screen_utils import capture_chessboard, find_chessboard_region

# --- Configuration ---
OUTPUT_DIR = Path("./data/empty_squares")
NUM_SAMPLES_PER_COLOR = 10  # Number of empty squares to save per color (light/dark)
SQUARE_DETECTION_THRESHOLD = 20

# --- Main Logic ---


def capture_and_save_empty_squares() -> None:
    """Finds the chessboard, captures it, extracts empty squares, and saves them."""

    # Load the CNN model first
    print("Loading CNN model...")
    model, transform, class_names = load_cnn_model()
    if model is None or transform is None or class_names is None:
        print("Error: Failed to load CNN model. Cannot find chessboard without it.")
        return

    print("Attempting to find the chessboard region...")
    region = find_chessboard_region(
        model=model,
        transform=transform,
        class_names=class_names,
    )
    if not region:
        print("Error: Chessboard region not found. Make sure it's visible on screen.")
        return

    print(f"Chessboard region found: {region}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    input(
        "Please ensure the chessboard is empty or has only a few pieces, then press Enter...",
    )

    print("Capturing chessboard image...")
    board_img = capture_chessboard(region=region)
    if not isinstance(board_img, Image.Image):
        print("Error: Failed to capture chessboard image.")
        return

    # Convert to grayscale for easier light/dark detection
    board_img_gray = board_img.convert("L")

    width, height = board_img.size
    square_width = width // 8
    square_height = height // 8

    print(f"Calculated square size: {square_width}x{square_height}")

    light_squares_saved = 0
    dark_squares_saved = 0

    print(f"Extracting and saving empty squares to {OUTPUT_DIR}...")

    for r in range(8):
        for c in range(8):
            if (
                light_squares_saved >= NUM_SAMPLES_PER_COLOR
                and dark_squares_saved >= NUM_SAMPLES_PER_COLOR
            ):
                break  # Stop once we have enough samples

            left = c * square_width
            top = r * square_height
            right = left + square_width
            bottom = top + square_height

            square_img = board_img.crop((left, top, right, bottom))
            square_img_gray = board_img_gray.crop((left, top, right, bottom))

            # Determine if the square is light or dark (simple average pixel value)
            avg_color = sum(square_img_gray.getdata()) / (square_width * square_height)
            print(f"  Square ({r},{c}): Avg Gray = {avg_color:.2f}")

            is_light = avg_color > 180  # Adjust this threshold if needed

            # Basic check if square seems empty (more sophisticated checks could be added)
            # For now, we rely on the user ensuring the board is mostly empty.
            is_empty = True  # Placeholder

            if is_empty:
                if is_light and light_squares_saved < NUM_SAMPLES_PER_COLOR:
                    filename = f"empty_light_{light_squares_saved:02d}.png"
                    square_img.save(OUTPUT_DIR / filename)
                    light_squares_saved += 1
                    print(f"Saved light square: {filename}")
                elif not is_light and dark_squares_saved < NUM_SAMPLES_PER_COLOR:
                    filename = f"empty_dark_{dark_squares_saved:02d}.png"
                    square_img.save(OUTPUT_DIR / filename)
                    dark_squares_saved += 1
                    print(f"Saved dark square: {filename}")

        if (
            light_squares_saved >= NUM_SAMPLES_PER_COLOR
            and dark_squares_saved >= NUM_SAMPLES_PER_COLOR
        ):
            break

    print("Finished saving empty squares.")
    print(
        f"Saved {light_squares_saved} light squares and {dark_squares_saved} dark squares.",
    )
    if (
        light_squares_saved < NUM_SAMPLES_PER_COLOR
        or dark_squares_saved < NUM_SAMPLES_PER_COLOR
    ):
        print("Warning: Could not find enough empty squares of both colors.")
        print("Make sure the board was sufficiently empty during capture.")


if __name__ == "__main__":
    capture_and_save_empty_squares()
