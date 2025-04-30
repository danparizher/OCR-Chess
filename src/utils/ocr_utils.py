from __future__ import annotations

from typing import TYPE_CHECKING

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from src.config import BOARD_SIZE, INPUT_SIZE
from src.models.cnn_definition import SimpleCNN

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path


# --- CNN Model Loading ---
def load_cnn_model(
    model_path: str | Path = "models/chess_piece_cnn.pth",
    num_classes: int = 13,
    input_size: int = INPUT_SIZE,
) -> tuple[torch.nn.Module | None, transforms.Compose | None, list[str] | None]:
    """Loads the trained CNN model and defines the necessary transforms."""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = SimpleCNN(num_classes=num_classes)
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Loaded trained model from {model_path}")
    except FileNotFoundError:
        print(f"Error: Model file not found at {model_path}. Train the model first.")
        return None, None, None
    except Exception as e:
        print(f"Error loading model state_dict: {e}")
        return None, None, None

    model.to(device)
    model.eval()  # Set model to evaluation mode

    # Define the same transforms used during validation
    # IMPORTANT: Ensure these match the validation transforms in train_cnn.py
    data_transform = transforms.Compose(
        [
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ],
    )

    # We need the class names in the order they were used during training
    # Typically derived from folder names by ImageFolder
    # IMPORTANT: Ensure this list matches the output of train_cnn.py
    class_names = [
        "bB",
        "bK",
        "bN",
        "bP",
        "bQ",
        "bR",
        "empty",
        "wB",
        "wK",
        "wN",
        "wP",
        "wQ",
        "wR",
    ]
    # TODO: Consider saving class_names list alongside the model during training
    #       or inferring it reliably from the data directory structure.
    if len(class_names) != num_classes:
        print(
            f"Warning: Number of class names ({len(class_names)}) does not match num_classes ({num_classes})",
        )

    return model, data_transform, class_names


# --- Internal CNN Square Prediction Helper ---
def _iter_square_predictions(
    pil_image: Image.Image,
    model: torch.nn.Module,
    transform: transforms.Compose,
    class_names: list[str],
) -> Iterator[tuple[int, int, str]]:
    """
    Iterates through board squares, performs CNN prediction, and yields results.

    Yields:
        tuple[int, int, str]: Row index, column index, and predicted class name.
    """
    w, h = pil_image.size
    if w == 0 or h == 0:
        return  # Stop iteration for empty images

    square_h = h / BOARD_SIZE
    square_w = w / BOARD_SIZE
    device = next(model.parameters()).device

    for i in range(BOARD_SIZE):
        for j in range(BOARD_SIZE):
            y1 = round(i * square_h)
            y2 = round((i + 1) * square_h)
            x1 = round(j * square_w)
            x2 = round((j + 1) * square_w)

            # Ensure coordinates are within image bounds
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            if x1 >= x2 or y1 >= y2:  # Skip if the square has zero area
                continue

            # Crop square from PIL image
            square_img_pil = pil_image.crop((x1, y1, x2, y2))

            # Apply transformations and predict
            try:
                input_tensor = transform(square_img_pil)
                assert isinstance(input_tensor, torch.Tensor)
                input_batch = input_tensor.unsqueeze(0).to(device)

                with torch.no_grad():
                    output = model(input_batch)
                    _, predicted_idx = torch.max(output, 1)

                predicted_class_index = int(predicted_idx.item())
                predicted_class = class_names[predicted_class_index]
                yield i, j, predicted_class
            except Exception as e:
                print(
                    f"Error processing square ({i},{j}) during prediction: {e}",
                )
                # Optionally yield an error marker or skip? Currently skips.


# --- CNN Piece Counting Helper ---
def count_pieces_in_region(
    image: Image.Image,
    model: torch.nn.Module,
    transform: transforms.Compose,
    class_names: list[str],
) -> int:
    """
    Counts the number of non-empty squares in a given image region using the CNN model.
    (Refactored to use _iter_square_predictions)
    """
    if not isinstance(image, Image.Image):
        print("Warning: count_pieces_in_region expects a PIL Image.")
        return 0

    pil_image = image.convert("RGB")
    return sum(
        predicted_class != "empty"
        for _, _, predicted_class in _iter_square_predictions(
            pil_image,
            model,
            transform,
            class_names,
        )
    )


# --- Board Extraction (CNN Version) ---


def extract_board_from_image(
    image_or_path: str | Image.Image | np.ndarray,
    model: torch.nn.Module,
    transform: transforms.Compose,
    class_names: list[str],
) -> list[list[str]]:
    """
    Given a path or image/array, use the trained CNN model to extract piece positions.
    Returns a 2D list representing the board (8x8).
    (Refactored to use _iter_square_predictions)
    """
    # Convert input to PIL Image (RGB) - Moved conversion logic here
    pil_image: Image.Image
    if isinstance(image_or_path, str):
        pil_image = Image.open(image_or_path).convert("RGB")
    elif isinstance(image_or_path, np.ndarray):
        # Assuming BGR from OpenCV capture, convert to RGB
        if image_or_path.shape[2] == 3:
            pil_image = Image.fromarray(cv2.cvtColor(image_or_path, cv2.COLOR_BGR2RGB))
        elif image_or_path.shape[2] == 4:
            pil_image = Image.fromarray(cv2.cvtColor(image_or_path, cv2.COLOR_BGRA2RGB))
        else:  # Grayscale? Handle appropriately or raise error
            pil_image = Image.fromarray(image_or_path).convert("RGB")
    elif isinstance(image_or_path, Image.Image):
        pil_image = image_or_path.convert("RGB")

    board = [["" for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]

    # Map class name to FEN representation (handle 'empty')
    piece_map = {
        "bB": "b",
        "bK": "k",
        "bN": "n",
        "bP": "p",
        "bQ": "q",
        "bR": "r",
        "wB": "B",
        "wK": "K",
        "wN": "N",
        "wP": "P",
        "wQ": "Q",
        "wR": "R",
        "empty": "",  # Map 'empty' class to empty string for the board
    }

    # Use the generator to get predictions and fill the board
    for i, j, predicted_class in _iter_square_predictions(
        pil_image,
        model,
        transform,
        class_names,
    ):
        board[i][j] = piece_map.get(
            predicted_class,
            "",
        )  # Default to empty if class not in map

    return board


def _pretty_print_board(board: list[list[str]]) -> str:
    """Formats the board list of lists into a readable string."""
    return "\n".join(" ".join(piece or "." for piece in row) for row in board)


def board_to_fen(board: list[list[str]], turn: str = "w") -> str:
    """
    Convert 2D board array (from extract_board_from_image) to FEN string.

    Raises:
        ValueError: If any row does not represent 8 squares (invalid FEN row).
        ValueError: If there is not exactly one white king and one black king.
    """
    # --- King count validation ---
    flat_board = [cell.strip() for row in board for cell in row]
    white_king_count = flat_board.count("K")
    black_king_count = flat_board.count("k")
    if white_king_count != 1 or black_king_count != 1:
        pretty_board = _pretty_print_board(board)
        msg = f"Invalid board: found {white_king_count} white kings and {black_king_count} black kings.\nBoard:\n{pretty_board}"
        raise ValueError(msg)
    # --- End king count validation ---
    fen_rows = []
    for row in board:
        fen_row = ""
        empty = 0
        for cell in row:
            cell_str = cell.strip()
            # treat empty (or ".") as blank square
            if cell_str in {"", "."}:
                empty += 1
            else:
                # only keep the first OCR'd piece-letter
                piece = cell_str[0]
                if empty > 0:
                    fen_row += str(empty)
                    empty = 0
                fen_row += piece

        if empty > 0:
            fen_row += str(empty)

        # make sure each row really is 8 squares
        total_squares = sum(int(ch) if ch.isdigit() else 1 for ch in fen_row)
        if total_squares != BOARD_SIZE:
            msg = f"Invalid FEN row {fen_row!r}: represents {total_squares} squares, expected {BOARD_SIZE}"
            raise ValueError(
                msg,
            )

        fen_rows.append(fen_row)

    fen = "/".join(fen_rows)
    return f"{fen} {turn} KQkq - 0 1"


def detect_player_color_and_orientation(board: list[list[str]]) -> tuple[str, bool]:
    """
    Determine player color ('w' or 'b') and board orientation.

    Returns:
        tuple[str, bool]: (turn, is_white_at_bottom)
        - turn: 'w' or 'b' (best guess based on king back-rank relative to orientation)
        - is_white_at_bottom: True if the board array likely represents
                              White pieces at the bottom (rows 4-7).
                              False if Black pieces are likely at the bottom.
    """
    white_king_pos = None
    black_king_pos = None
    for r, row in enumerate(board):
        for c, piece in enumerate(row):
            if piece == "K":
                white_king_pos = (r, c)
            elif piece == "k":
                black_king_pos = (r, c)

    if white_king_pos is None or black_king_pos is None:
        # If kings are missing, cannot determine orientation or turn reliably
        print(
            "Warning: Could not find both kings. Defaulting to White's turn, White-at-bottom.",
        )
        return "w", True

    w_row, _ = white_king_pos
    b_row, _ = black_king_pos

    # --- Determine Orientation --- based on relative king positions
    is_white_at_bottom = True  # Default assumption
    if w_row >= 4 and b_row < 4:
        is_white_at_bottom = True
    elif b_row >= 4 and w_row < 4:
        is_white_at_bottom = False
    else:
        # Use piece count heuristic when king positions are ambiguous
        is_white_at_bottom = _guess_orientation_by_piece_count(
            board,
            white_king_pos,
            black_king_pos,
        )
    # --- Determine Turn --- based on orientation and back-rank check
    turn = "w"  # Default turn
    log_msg = ""
    if is_white_at_bottom:
        # White at bottom: W expected on row 7 (rank 1), B on row 0 (rank 8)
        w_on_back = w_row == 7
        b_on_back = b_row == 0
        turn = (
            "w"
            if (
                (w_on_back and not b_on_back)
                or ((not b_on_back or w_on_back) and b_on_back)
                or not b_on_back
            )
            else "b"
        )
    else:
        # Black at bottom: B expected on row 7 (rank 8), W on row 0 (rank 1)
        b_on_back = b_row == 7
        w_on_back = w_row == 0
        if (
            (b_on_back and not w_on_back)
            or ((not w_on_back or b_on_back) and w_on_back)
            or not w_on_back
        ):
            turn = "b"
        else:
            turn = "w"
    print(log_msg)
    return turn, is_white_at_bottom


def _guess_orientation_by_piece_count(
    board: list[list[str]],
    white_king_pos: tuple[int, int],
    black_king_pos: tuple[int, int],
) -> bool:
    """
    Guess board orientation based on piece counts when king positions are ambiguous.

    Returns:
        bool: True if White is likely at the bottom, False otherwise.
    """
    # Ambiguous: Kings on same side, or one/both exactly on midline.
    # Using piece count heuristic as tie-breaker.
    white_top_count = 0
    black_top_count = 0
    white_bottom_count = 0
    black_bottom_count = 0

    for r, row_pieces in enumerate(board):
        for piece in row_pieces:
            if r < 4:  # Top half
                if piece.isupper():
                    white_top_count += 1
                elif piece.islower():
                    black_top_count += 1
            elif piece.isupper():
                white_bottom_count += 1
            elif piece.islower():
                black_bottom_count += 1

    if white_bottom_count > black_bottom_count:
        result = True
        print(
            f"Info: Ambiguous kings (W: {white_king_pos}, B: {black_king_pos}). "
            f"Orientation guess (White at bottom) based on piece count "
            f"(Bottom: W={white_bottom_count}, B={black_bottom_count}; Top: W={white_top_count}, B={black_top_count}).",
        )
    elif black_bottom_count > white_bottom_count:
        result = False
        print(
            f"Info: Ambiguous kings (W: {white_king_pos}, B: {black_king_pos}). "
            f"Orientation guess (Black at bottom) based on piece count "
            f"(Bottom: W={white_bottom_count}, B={black_bottom_count}; Top: W={white_top_count}, B={black_top_count}).",
        )
    else:  # Counts are equal, stick to default
        result = True
        print(
            f"Warning: Ambiguous king positions (W: {white_king_pos}, B: {black_king_pos}) "
            f"and piece counts are equal (Bottom: W={white_bottom_count}, B={black_bottom_count}; Top: W={white_top_count}, B={black_top_count}). Defaulting to White-at-bottom.",
        )

    return result


def flip_board(board: list[list[str]]) -> list[list[str]]:
    """Flips the board vertically and horizontally."""
    return [row[::-1] for row in board[::-1]]
