# config.py

from pathlib import Path

# Board size (number of squares per side)
BOARD_SIZE = 8

# Input image size for CNN (must match training)
INPUT_SIZE = 64

# Chess Engine Configuration
ENGINE_THREADS = 10  # Number of physical CPU cores
ENGINE_HASH_MB = 2048  # Depends on available RAM

# Path Configuration
PROJECT_ROOT = Path(__file__).parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODELS_DIR / "chess_piece_cnn.pth"
BIN_DIR = PROJECT_ROOT / "bin"
