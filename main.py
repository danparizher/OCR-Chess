from __future__ import annotations

import concurrent.futures
import contextlib
import os
import traceback
from typing import TYPE_CHECKING

import chess.engine
import xxhash
from PIL import Image
from PyQt6 import QtCore

from chess_engine import ChessEngine
from ocr_utils import (
    board_to_fen,
    detect_player_color,
    extract_board_from_image,
    flip_board,
    load_piece_templates,
)
from overlay import show_overlay
from screen_utils import (
    capture_chessboard,
    find_chessboard_region,
)

if TYPE_CHECKING:
    import numpy as np

os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "1"


def flip_move(move: str) -> str:
    """
    Flip a move string (e.g., 'e2e4') for black's perspective.
    'a1' <-> 'h8', 'b1' <-> 'g8', ..., 'h1' <-> 'a8', etc.
    """

    def flip_square(sq: str) -> str:
        file = sq[0]
        rank = sq[1]
        flipped_file = chr(ord("h") - (ord(file) - ord("a")))
        flipped_rank = str(9 - int(rank))
        return flipped_file + flipped_rank

    return flip_square(move[:2]) + flip_square(move[2:4])


def ocr_and_engine_task(
    board_img: Image.Image,
    templates: dict[str, np.ndarray],
    preprocessed_templates: dict[str, np.ndarray],
) -> tuple[str | None, bool]:
    board = extract_board_from_image(
        board_img,
        templates=templates,
        preprocessed_templates=preprocessed_templates,
    )
    turn = detect_player_color(board)
    is_black = turn == "b"
    if is_black:
        board = flip_board(board)
    try:
        fen = board_to_fen(board, turn=turn)
    except ValueError as e:
        print(f"[OCR/Engine] Board to FEN error: {e}")
        fen = None
    return fen, is_black


def main() -> None:
    region = find_chessboard_region(debug=True)
    print("Detected chessboard region:", region)
    left, top, right, bottom = region
    width = right - left
    height = bottom - top
    region_xywh = (left, top, width, height)

    app, overlay = show_overlay(region_xywh)

    engine = ChessEngine()
    last_fen = None
    last_img_hash = None  # Store hash of last board image

    # Load templates once
    templates, preprocessed_templates = load_piece_templates()

    # Use ProcessPoolExecutor for CPU-bound OCR/engine work
    executor = concurrent.futures.ProcessPoolExecutor(max_workers=1)
    future = None

    def fast_image_hash(img: Image.Image) -> str:
        # Downscale to 32x32 and hash raw bytes for speed
        small = img.resize((32, 32)).tobytes()
        return xxhash.xxh64(small).hexdigest()

    def update_overlay() -> None:
        nonlocal engine, last_fen, future, last_img_hash
        try:
            # If a previous OCR/engine task finished, process its result
            if future is not None and future.done():
                try:
                    result = future.result()
                    if not isinstance(result, tuple) or len(result) != 2:
                        print("Unexpected result from OCR/engine task:", result)
                        future = None
                        return
                    fen, is_black = result  # type: ignore
                    if fen is not None and fen != last_fen:
                        print("FEN changed:", fen)
                        try:
                            best_move = engine.get_best_move(fen)
                            print("Best move:", best_move)
                            if best_move:
                                move_to_show = (
                                    flip_move(best_move) if is_black else best_move
                                )
                                overlay.update_move(str(move_to_show))
                                last_fen = fen
                        except chess.engine.EngineTerminatedError:
                            print("Invalid FEN detected, restarting")
                            with contextlib.suppress(Exception):
                                engine.close()
                            engine = ChessEngine()
                        except Exception as e:
                            print(
                                "Error while getting best move (will retry next tick):",
                                e,
                            )
                            traceback.print_exc()
                except Exception as e:
                    print("Error in OCR/engine task (will retry next tick):", e)
                    traceback.print_exc()
                future = None
            # If no OCR/engine task is running, start a new one
            if future is None or future.done():
                board_img = capture_chessboard(region=region)
                if not isinstance(board_img, Image.Image):
                    print("capture_chessboard() did not return a PIL Image")
                    return
                # Compute fast hash of the image to detect changes
                img_hash = fast_image_hash(board_img)
                if img_hash == last_img_hash:
                    # No change in board image, skip OCR/engine work
                    return
                last_img_hash = img_hash
                future = executor.submit(
                    ocr_and_engine_task,
                    board_img,
                    templates,
                    preprocessed_templates,
                )
        except Exception as e:
            print("Error during board capture/OCR (will retry next tick):", e)
            traceback.print_exc()

    timer = QtCore.QTimer()
    timer.timeout.connect(update_overlay)
    timer.start(1)

    app.exec()
    engine.close()
    executor.shutdown()


if __name__ == "__main__":
    main()
