from __future__ import annotations

import contextlib
import logging

import chess
import chess.engine

from src import config

logger = logging.getLogger(__name__)


class ChessEngine:
    def __init__(
        self,
        threads: int | None = None,
        hash_mb: int | None = None,
    ) -> None:
        """
        Initialize the engine using Stockfish.

        Args:
            threads: Number of engine threads to use. Defaults to config.ENGINE_THREADS.
            hash_mb: Hash memory in MB. Defaults to config.ENGINE_HASH_MB.
        """
        engine_threads = threads if threads is not None else config.ENGINE_THREADS
        engine_hash_mb = hash_mb if hash_mb is not None else config.ENGINE_HASH_MB

        try:
            self.engine = chess.engine.SimpleEngine.popen_uci(
                r"bin/stockfish-windows-x86-64-avx2.exe",
            )
            # Configure engine options
            self.engine.configure({"Threads": engine_threads, "Hash": engine_hash_mb})
        except Exception:
            logger.exception("Failed to initialize chess engine")
            raise

    def get_best_move(self, fen: str, time_limit: float = 0) -> str | None:
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

    def get_evaluation(
        self,
        fen: str,
        time_limit: float = 0.0,
    ) -> chess.engine.InfoDict | None:
        """
        Get the evaluation score for the current board position.

        Returns:
            A dictionary containing evaluation info, or None if an error occurs.
            See chess.engine.InfoDict for details.
        """
        try:
            board = chess.Board(fen)
            return self.engine.analyse(board, chess.engine.Limit(time=time_limit))
        except chess.engine.EngineTerminatedError:
            logger.error("Engine terminated unexpectedly during evaluation")  # noqa: TRY400
            raise
        except Exception:
            logger.exception("Error getting board evaluation")
            return None

    def close(self) -> None:
        with contextlib.suppress(Exception):
            self.engine.quit()
