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

from src.engine.chess_engine import ChessEngine
from src.overlay import show_overlay
from src.utils.ocr_utils import (
    board_to_fen,
    detect_player_color_and_orientation,
    extract_board_from_image,
    flip_board,
    load_cnn_model,
)
from src.utils.screen_utils import (
    capture_chessboard,
    find_chessboard_region,
)

if TYPE_CHECKING:
    import torch
    from torchvision import transforms

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
    model: torch.nn.Module,
    transform: transforms.Compose,
    class_names: list[str],
) -> tuple[str | None, bool]:
    board_array_capture = extract_board_from_image(
        board_img,
        model=model,
        transform=transform,
        class_names=class_names,
    )

    # Detect turn AND the orientation of the captured board
    turn, is_white_at_bottom_capture = detect_player_color_and_orientation(
        board_array_capture,
    )

    # --- Standardize board orientation for FEN --- Always White-at-bottom
    board_for_fen = board_array_capture
    if not is_white_at_bottom_capture:
        # If capture had Black at bottom, flip it to White-at-bottom for FEN
        board_for_fen = flip_board(board_array_capture)

    try:
        # Generate FEN using the standardized (White-at-bottom) board and detected turn
        fen = board_to_fen(board_for_fen, turn=turn)
    except ValueError as e:
        print(f"[OCR/Engine] Board to FEN error: {e}")
        fen = None

    # Return the generated FEN and the ORIGINAL orientation of the capture
    return fen, is_white_at_bottom_capture


def main() -> None:
    # Load CNN Model once
    print("Loading CNN model...")
    model, transform, class_names = load_cnn_model()
    if model is None or transform is None or class_names is None:
        print("Failed to load CNN model. Exiting.")
        return

    # --- Find Chessboard Region (using CNN verification) ---
    print("Finding chessboard region...")
    region = find_chessboard_region(
        model=model,
        transform=transform,
        class_names=class_names,
    )
    # Handle case where board is not found
    if region is None:
        print("Chessboard region not found. Exiting.")
        return
    print("Detected chessboard region:", region)
    left, top, right, bottom = region
    width = right - left
    height = bottom - top
    region_xywh = (left, top, width, height)

    app, overlay = show_overlay(region_xywh)

    engine = ChessEngine()
    last_fen = None
    last_img_hash = None  # Store hash of last board image

    # Use ProcessPoolExecutor for CPU-bound OCR/engine work
    # NOTE: Passing model directly might be inefficient if it's large.
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
                    # Ensure result is a tuple of (str | None, bool)
                    if (
                        not isinstance(result, tuple)
                        or len(result) != 2
                        or not isinstance(result[1], bool)
                    ):
                        print("Unexpected result type from OCR/engine task:", result)
                        future = None
                        return

                    fen, is_white_at_bottom_capture = result  # type: ignore

                    if fen is not None and fen != last_fen:
                        print("FEN changed:", fen)
                        last_fen = fen
                        try:
                            # Get evaluation first
                            eval_info = engine.get_evaluation(fen)
                            if eval_info and "score" in eval_info:
                                pov_score: chess.engine.PovScore = eval_info["score"]

                                # Determine whose turn it is from the FEN
                                turn = fen.split()[1]  # 'w' or 'b'
                                score: chess.engine.Score | None = None

                                score = (
                                    pov_score.white()
                                    if turn == "w"
                                    else pov_score.black()
                                )
                                # Convert PovScore to a readable format
                                if score.is_mate():
                                    mate_moves = score.mate()
                                    if mate_moves is not None:
                                        # Display mate relative to White (+M5 = White mates, -M3 = Black mates)
                                        final_mate = (
                                            mate_moves if turn == "w" else -mate_moves
                                        )
                                        print(f"Evaluation: Mate in {final_mate}")
                                    else:
                                        print("Evaluation: Mate (Unknown moves)")
                                else:
                                    # Get centipawn score relative to the current player
                                    cp = score.score(mate_score=10000)
                                    if cp is not None:
                                        if turn == "b":
                                            cp = -cp
                                        pawn_score = cp / 100.0
                                        print(f"Evaluation: {pawn_score:+.2f}")
                                    else:
                                        print("Evaluation: N/A (score is None)")
                            else:
                                print("Evaluation: N/A (no score info)")

                            best_move = engine.get_best_move(fen)
                            print("Best move (engine standard):", best_move)
                            if best_move:
                                needs_flip_for_display = not is_white_at_bottom_capture
                                move_to_show = (
                                    flip_move(best_move)
                                    if needs_flip_for_display
                                    else best_move
                                )
                                overlay.update_move(str(move_to_show))
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
                # Submit task with model, transform, and class_names
                future = executor.submit(
                    ocr_and_engine_task,
                    board_img,
                    model,
                    transform,
                    class_names,
                )
        except Exception as e:
            print("Error during board capture/OCR (will retry next tick):", e)
            traceback.print_exc()

    timer = QtCore.QTimer()
    timer.timeout.connect(update_overlay)
    timer.start(250)

    app.exec()
    engine.close()
    executor.shutdown()


if __name__ == "__main__":
    main()
