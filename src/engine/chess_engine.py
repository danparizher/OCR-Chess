from __future__ import annotations

import contextlib
import logging

import chess
import chess.engine

logger = logging.getLogger(__name__)


class ChessEngine:
    def __init__(self) -> None:
        """
        Initialize the engine using Stockfish.
        Provide the path to the Stockfish executable.
        """
        try:
            # Replace "path/to/stockfish" with the actual path to the Stockfish binary
            self.engine = chess.engine.SimpleEngine.popen_uci(
                r"bin/stockfish-windows-x86-64-avx2.exe",
            )
            logger.info("Chess engine initialized successfully.")
        except Exception:
            logger.exception("Failed to initialize chess engine")
            raise

    def get_best_move(self, fen: str, time_limit: float = 0.5) -> str | None:
        """
        Given a FEN string, return the best move in UCI format.

        Raises:
            chess.engine.EngineTerminatedError: If the engine terminates unexpectedly.
        """
        try:
            board = chess.Board(fen)
            result = self.engine.play(board, chess.engine.Limit(time=time_limit))
            if result.move is not None and result.move.uci() is not None:
                return result.move.uci()
            return None
        except chess.engine.EngineTerminatedError:
            logger.error("Engine terminated unexpectedly")  # noqa: TRY400
            raise
        except Exception:
            logger.exception("Error getting best move")
            raise

    def close(self) -> None:
        with contextlib.suppress(Exception):
            self.engine.quit()
