from __future__ import annotations

import cv2
import numpy as np
from PIL import Image, ImageGrab


def capture_chessboard(region: tuple[int, int, int, int] | None = None) -> Image.Image:
    """
    Capture the chessboard area from the screen.
    If region is None, capture the entire screen.
    Returns a PIL Image.
    """
    if region is None:
        # grab entire screen
        return ImageGrab.grab()
    # grab just the given bbox
    return ImageGrab.grab(bbox=region)


def find_chessboard_region(debug: bool = False) -> tuple[int, int, int, int]:
    """
    Automatically detect the chessboard region on the screen.
    Returns (left, top, right, bottom) bounding box.
    If debug is True, saves a debug image with the detected region.
    """
    # Capture the whole screen
    screen = ImageGrab.grab()
    screen_np = np.array(screen)
    img = cv2.cvtColor(screen_np, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)

    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max_area = 0
    best_rect = (0, 0, img.shape[1], img.shape[0])
    for cnt in contours:
        approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)
        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(approx)
            area = w * h
            aspect = w / h if h != 0 else 0
            # Heuristic: look for large, nearly square regions
            if area > max_area and 0.8 < aspect < 1.2 and w > 100 and h > 100:
                max_area = area
                best_rect = (x, y, x + w, y + h)
    if debug:
        debug_img = img.copy()
        x1, y1, x2, y2 = best_rect
        cv2.rectangle(debug_img, (x1, y1), (x2, y2), (0, 255, 0), 3)
        cv2.imwrite("test_output/chessboard_detected.png", debug_img)
    return best_rect
