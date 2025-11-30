from __future__ import annotations

import contextlib
import logging
import platform
from pathlib import Path

import chess
import chess.engine

from src import config

logger = logging.getLogger(__name__)


def _get_stockfish_path() -> Path:
    """Get the Stockfish binary path based on the current platform."""
    system = platform.system().lower()
    machine = platform.machine().lower()

    if system == "darwin":
        return (
            config.BIN_DIR / "stockfish-macos-arm64"
            if "arm64" in machine or "aarch64" in machine
            else config.BIN_DIR / "stockfish-macos-x86-64"
        )
    elif system == "linux":
        return (
            config.BIN_DIR / "stockfish-linux-arm64"
            if "arm64" in machine or "aarch64" in machine
            else config.BIN_DIR / "stockfish-linux-x86-64"
        )
    elif system == "windows":
        if "x86_64" in machine or "amd64" in machine:
            return config.BIN_DIR / "stockfish-windows-x86-64-avx2.exe"
        elif "x86" in machine or "i386" in machine or "i686" in machine:
            return config.BIN_DIR / "stockfish-windows-x86-32.exe"
        else:
            # Fallback for Windows
            return config.BIN_DIR / "stockfish-windows-x86-64-avx2.exe"
    else:
        # Fallback: try to find any stockfish executable
        stockfish_exe = "stockfish.exe" if system == "windows" else "stockfish"
        return config.BIN_DIR / stockfish_exe


class ChessEngine:
    def __init__(
        self,
        threads: int | None = None,
        hash_mb: int | None = None,
        engine_path: str | Path | None = None,
    ) -> None:
        """
        Initialize the engine using Stockfish.

        Args:
            threads: Number of engine threads to use. Defaults to config.ENGINE_THREADS.
            hash_mb: Hash memory in MB. Defaults to config.ENGINE_HASH_MB.
            engine_path: Path to Stockfish binary. If None, auto-detects based on platform.
        """
        engine_threads = threads if threads is not None else config.ENGINE_THREADS
        engine_hash_mb = hash_mb if hash_mb is not None else config.ENGINE_HASH_MB

        if engine_path is None:
            engine_path = _get_stockfish_path()
        else:
            engine_path = Path(engine_path)

        if not engine_path.exists():
            msg = (
                f"Stockfish binary not found at {engine_path}. Please ensure it exists."
            )
            logger.error(msg)
            raise FileNotFoundError(msg)

        try:
            self.engine = chess.engine.SimpleEngine.popen_uci(str(engine_path))
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
