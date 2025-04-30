from __future__ import annotations

from typing import TYPE_CHECKING

import cv2
import numpy as np
from PIL import Image, ImageGrab

from src.utils.ocr_utils import count_pieces_in_region

if TYPE_CHECKING:
    import torch
    from torchvision import transforms


def capture_chessboard(region: tuple[int, int, int, int] | None = None) -> Image.Image:
    """
    Capture the chessboard area from the screen.
    If region is None, capture the entire screen.
    Returns a PIL Image.
    """
    return ImageGrab.grab() if region is None else ImageGrab.grab(bbox=region)


def find_chessboard_region(
    model: torch.nn.Module,
    transform: transforms.Compose,
    class_names: list[str],
) -> tuple[int, int, int, int] | None:
    """
    Automatically detect the chessboard region using contours and CNN verification.
    Generates candidate regions using geometry, then verifies using piece count from CNN.
    Returns (left, top, right, bottom) bounding box or None if not found.
    """
    # Capture the whole screen
    screen_pil = ImageGrab.grab()  # Keep PIL image for cropping
    screen_np = np.array(screen_pil)
    img_cv = cv2.cvtColor(screen_np, cv2.COLOR_RGB2BGR)  # Keep CV image for contours

    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)

    # Find contours using hierarchy
    contours, hierarchy = cv2.findContours(
        edges,
        cv2.RETR_TREE,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    candidate_rects = []  # Store potential board rectangles (x1, y1, x2, y2)

    # Ensure hierarchy is valid
    if hierarchy is None or len(hierarchy.shape) != 3 or hierarchy.shape[0] != 1:
        print("Warning: Unexpected hierarchy format from findContours.")
        hierarchy = np.array([[[-1, -1, -1, -1]]])

    # --- Candidate Generation (Geometric) ---
    for i, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        if area < 5000:  # Increase min area threshold slightly
            continue

        approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)

        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(approx)
            aspect = w / h if h != 0 else 0

            # Relax aspect ratio slightly, keep size constraint
            if 0.85 < aspect < 1.15 and w > 150 and h > 150:
                # Check if it's a top-level contour
                is_top_level = hierarchy[0][i][3] == -1
                if is_top_level:
                    candidate_rects.append((x, y, x + w, y + h))

    if not candidate_rects:
        print("No geometric candidates found for chessboard.")
        return None

    # --- Candidate Verification (CNN) ---
    best_rect = None
    max_pieces = -1  # Use -1 to ensure the first valid count becomes max
    min_required_pieces = 4  # Require at least a few pieces to be detected

    for rect in candidate_rects:
        x1, y1, x2, y2 = rect
        # Crop the original PIL image
        candidate_img_pil = screen_pil.crop((x1, y1, x2, y2))

        # Count pieces in the cropped region
        piece_count = count_pieces_in_region(
            candidate_img_pil,
            model,
            transform,
            class_names,
        )

        # Select the candidate with the most pieces (above threshold)
        if piece_count > max_pieces and piece_count >= min_required_pieces:
            max_pieces = piece_count
            best_rect = rect

    print(f"Selected board: {best_rect} with {max_pieces} pieces.")

    return best_rect
