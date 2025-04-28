from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytesseract
from PIL import Image

from config import BOARD_SIZE, TESSERACT_CMD

pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD


def load_piece_templates(
    template_folder: str = "python-chess-pieces",
    size: tuple[int, int] = (128, 128),
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """
    Load all piece templates from the given folder as color images (no preprocessing).
    Handles transparency by replacing transparent pixels with white.
    Returns a tuple of dicts: (original_templates, preprocessed_templates)
    {piece_letter: template_image}, {piece_letter: preprocessed_template}

    Raises:
        FileNotFoundError: If a template image cannot be loaded.
    """
    piece_map = {
        "wK.png": "K",
        "wQ.png": "Q",
        "wR.png": "R",
        "wB.png": "B",
        "wN.png": "N",
        "wP.png": "P",
        "bK.png": "k",
        "bQ.png": "q",
        "bR.png": "r",
        "bB.png": "b",
        "bN.png": "n",
        "bP.png": "p",
    }
    templates = {}
    preprocessed_templates = {}
    for path in Path(template_folder).iterdir():
        filename = path.name
        if filename in piece_map:
            # Load with alpha channel
            img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            if img is None:
                msg = f"Could not load image at {filename}"
                raise FileNotFoundError(msg)
            # If image has alpha channel, replace transparent pixels with white
            if img.shape[2] == 4:
                alpha = img[:, :, 3]
                white_bg = np.ones_like(img[:, :, :3], dtype=np.uint8) * 255
                mask = alpha == 0
                img_rgb = img[:, :, :3].copy()
                img_rgb[mask] = white_bg[mask]
            else:
                img_rgb = img
            # Convert to BGR (if not already)
            if img_rgb.shape[2] == 3:
                img_bgr = (
                    cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
                    if img_rgb.shape[2] == 3
                    else img_rgb
                )
            else:
                img_bgr = img_rgb
            img_bgr = cv2.resize(img_bgr, size)
            templates[piece_map[filename]] = img_bgr
            # Preprocess template: grayscale, CLAHE, blur, Canny
            tmpl_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            tmpl_gray = clahe.apply(tmpl_gray)
            tmpl_gray = cv2.GaussianBlur(tmpl_gray, (3, 3), 0)
            tmpl_edges = cv2.Canny(tmpl_gray, 50, 150)
            preprocessed_templates[piece_map[filename]] = tmpl_edges
    return templates, preprocessed_templates


def match_piece(
    square_img: np.ndarray,
    preprocessed_templates: dict[str, np.ndarray],
) -> str:
    """
    Match the square image against all preprocessed templates (edge maps).
    Returns the best-matching piece or "" if below threshold.
    """
    best_piece = ""
    best_score = -1.0
    primary_scores = {}
    # Preprocess square: grayscale, CLAHE, blur, Canny
    square_gray = cv2.cvtColor(square_img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    square_gray = clahe.apply(square_gray)
    square_gray = cv2.GaussianBlur(square_gray, (3, 3), 0)
    square_edges = cv2.Canny(square_gray, 50, 150)
    for piece, tmpl_edges in preprocessed_templates.items():
        if square_edges.shape != tmpl_edges.shape:
            square_edges_resized = cv2.resize(
                square_edges,
                (tmpl_edges.shape[1], tmpl_edges.shape[0]),
            )
        else:
            square_edges_resized = square_edges
        res = cv2.matchTemplate(square_edges_resized, tmpl_edges, cv2.TM_CCOEFF_NORMED)
        _min_val, max_val, _min_loc, _max_loc = cv2.minMaxLoc(res)
        primary_scores[piece] = max_val
        if max_val > best_score:
            best_score = float(max_val)
            best_piece = piece
    if primary_scores:
        best_piece = max(primary_scores, key=lambda k: primary_scores[k])
        best_score = primary_scores[best_piece]
    else:
        best_piece = ""
        best_score = -1.0
    if all(abs(score) < 1e-6 for score in primary_scores.values()) or all(
        score < 0 for score in primary_scores.values()
    ):
        return ""
    return best_piece


def match_piece_advanced(
    square_img: np.ndarray,
    templates: dict[str, np.ndarray],
    preprocessed_templates: dict[str, np.ndarray],
    alpha: float = 0.7,
) -> str:
    """
    Match the square image against all templates using a combination of edge and grayscale matching.
    Processes original size square, then resizes features.
    Returns the best-matching piece or "" if below threshold.
    """
    square_gray = cv2.cvtColor(square_img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    square_gray_eq = clahe.apply(square_gray)
    square_blur = cv2.GaussianBlur(square_gray_eq, (3, 3), 0)
    square_edges = cv2.Canny(square_blur, 50, 150)

    target_size = (128, 128)

    if square_edges.shape[:2] != target_size:
        square_edges_resized = cv2.resize(
            square_edges,
            target_size,
            interpolation=cv2.INTER_NEAREST,
        )
    else:
        square_edges_resized = square_edges

    if square_blur.shape[:2] != target_size:
        square_blur_resized = cv2.resize(
            square_blur,
            target_size,
            interpolation=cv2.INTER_CUBIC,
        )
    else:
        square_blur_resized = square_blur

    best_piece = ""
    best_score = -float("inf")
    scores = {}

    for piece, tmpl in templates.items():
        # Grayscale match (comparing resized input blur to template blur)
        tmpl_gray = cv2.cvtColor(tmpl, cv2.COLOR_BGR2GRAY)
        tmpl_gray_eq = clahe.apply(tmpl_gray)
        tmpl_blur = cv2.GaussianBlur(tmpl_gray_eq, (3, 3), 0)
        res_gray = cv2.matchTemplate(
            square_blur_resized,
            tmpl_blur,
            cv2.TM_CCOEFF_NORMED,
        )
        _, max_val_gray, _, _ = cv2.minMaxLoc(res_gray)

        # Edge match (comparing resized input edges to template edges)
        tmpl_edges = preprocessed_templates[piece]

        # The check below should ideally not be needed now, but keep as safety
        if square_edges_resized.shape[:2] != tmpl_edges.shape[:2]:
            square_edges_final = cv2.resize(
                square_edges_resized,
                (tmpl_edges.shape[1], tmpl_edges.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
        else:
            square_edges_final = square_edges_resized

        res_edge = cv2.matchTemplate(
            square_edges_final,
            tmpl_edges,
            cv2.TM_CCOEFF_NORMED,
        )
        _, max_val_edge, _, _ = cv2.minMaxLoc(res_edge)

        # Combine scores
        score = alpha * max_val_edge + (1 - alpha) * max_val_gray
        scores[piece] = score
        if score > best_score:
            best_score = score
            best_piece = piece

    score_threshold = 0.1
    return "" if best_score < score_threshold else best_piece


def extract_board_from_image(
    image_or_path: str | Image.Image | np.ndarray,
    template_folder: str = "python-chess-pieces",
    use_template_matching: bool = True,
    templates: dict[str, np.ndarray] | None = None,
    preprocessed_templates: dict[str, np.ndarray] | None = None,
) -> list[list[str]]:
    """
    Given a path or image/array, use template matching to extract piece positions.
    Returns a 2D list representing the board (8x8).
    """
    # Accepts either a file path, PIL Image, or numpy array
    if isinstance(image_or_path, str):
        image = Image.open(image_or_path)
        img_cv = np.array(image)
        img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)
    elif isinstance(image_or_path, Image.Image):
        img_cv = np.array(image_or_path)
        img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)
    elif isinstance(image_or_path, np.ndarray):
        img_cv = image_or_path
        if img_cv.shape[2] == 4:
            img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGBA2BGR)
        elif img_cv.shape[2] == 3:
            img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)

    h, w = img_cv.shape[:2]
    square_h = h / BOARD_SIZE
    square_w = w / BOARD_SIZE
    board = [["" for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]

    # Use provided templates or load if not given
    if templates is None or preprocessed_templates is None:
        templates, preprocessed_templates = load_piece_templates(
            template_folder=template_folder,
            size=(128, 128),
        )

    for i in range(BOARD_SIZE):
        for j in range(BOARD_SIZE):
            y1 = round(i * square_h)
            y2 = round((i + 1) * square_h)
            x1 = round(j * square_w)
            x2 = round((j + 1) * square_w)
            y1_crop = max(y1, 0)
            y2_crop = min(y2, h)
            x1_crop = max(x1, 0)
            x2_crop = min(x2, w)
            square = img_cv[y1_crop:y2_crop, x1_crop:x2_crop]
            piece = (
                match_piece_advanced(
                    square,
                    templates,
                    preprocessed_templates,
                    alpha=0.7,
                )
                if use_template_matching
                else ""
            )
            board[i][j] = piece
    return board


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
        msg = f"Invalid board: found {white_king_count} white kings and {black_king_count} black kings. Board: {board}"
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


def detect_player_color(board: list[list[str]]) -> str:
    """
    Detect if the player is white or black based on the distribution of pieces
    on the board. Returns 'w' if white's pieces are predominantly on the
    bottom half, 'b' if black's pieces are predominantly on the bottom half.
    Falls back to king position if distribution is unclear.
    """
    white_pieces = {"K", "Q", "R", "B", "N", "P"}
    black_pieces = {"k", "q", "r", "b", "n", "p"}

    white_bottom_count = 0
    white_top_count = 0
    black_bottom_count = 0
    black_top_count = 0

    for i, row in enumerate(board):
        for piece in row:
            if piece in white_pieces:
                if i >= BOARD_SIZE / 2:
                    white_bottom_count += 1
                else:
                    white_top_count += 1
            elif piece in black_pieces:
                if i >= BOARD_SIZE / 2:
                    black_bottom_count += 1
                else:
                    black_top_count += 1

    # Check if bottom half has more white than black AND top half has more black than white
    if white_bottom_count > black_bottom_count and black_top_count > white_top_count:
        return "w"

    # Check if bottom half has more black than white AND top half has more white than black
    if black_bottom_count > white_bottom_count and white_top_count > black_top_count:
        print("[detect_player_color] Detected black based on piece distribution")
        return "b"

    # Fallback: Check king positions if distribution is ambiguous
    for i, row in enumerate(board):
        for piece in row:
            if piece == "K":
                return _get_color_from_king_pos_and_log(
                    "w",
                    i,
                    "b",
                    " based on white king position",
                )
            if piece == "k":
                return _get_color_from_king_pos_and_log(
                    "b",
                    i,
                    "w",
                    " based on black king position",
                )
    print("[detect_player_color] No clear detection, defaulting to white")
    # Default to white if still unclear (should be rare)
    return "w"


def _get_color_from_king_pos_and_log(
    primary_color: str,
    i: int,
    secondary_color: str,
    log_suffix: str,
) -> str:
    result = primary_color if i >= BOARD_SIZE / 2 else secondary_color
    print(f"[detect_player_color] Detected {result}{log_suffix}")
    return result


def flip_board(board: list[list[str]]) -> list[list[str]]:
    """
    Flip the board 180 degrees (for black's perspective).
    """
    return [row[::-1] for row in board[::-1]]
