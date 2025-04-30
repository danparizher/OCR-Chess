# OCR Chess - Computer Vision Chess Assistant

**Disclaimer & Ethical Use**: This project is developed solely for educational purposes, aiming to explore computer vision techniques (like screen scraping and template matching), GUI development (with PyQt6), and chess engine integration (Stockfish). It is **NOT** intended for use in any way that violates the terms of service of chess platforms or compromises the principles of fair play.

- **No Automation**: The tool intentionally lacks features for automated mouse control or direct interaction with chess interfaces. Including such capabilities would facilitate cheating, which I strongly discourage.
- **Educational Focus**: Use this project to learn about the underlying technologies, analyze your own past games (e.g., from screenshots or recordings), or experiment with computer vision on chess positions.
- **Prohibited Use**: Do **NOT** use this software to gain an unfair advantage in live games (online or over-the-board), tournaments, or any rated matches. Respect the rules of chess communities and platforms.

By using this software, you agree to employ it responsibly and ethically, strictly for learning and analysis purposes.

## Key Features

- **Pure Computer Vision Approach**: Analyzes chess positions from any screen source without relying on browser automation or specific website integrations.
- **Versatile Compatibility**:
  - Works with Chess960 (Fischer Random Chess) and other 8x8 variants.
  - Handles digital interfaces, offline games, and even images of physical boards.
- **Automatic Board & Piece Detection**: Identifies chessboard regions and recognizes pieces using a Convolutional Neural Network (CNN).
- **FEN Generation & Analysis**: Converts positions to FEN and uses the Stockfish engine for best move suggestions.
- **Visual Overlay**: Displays suggested moves directly on the screen.
- **Perspective Handling**: Works for both white and black perspectives.

## Requirements

- Python 3.13
- PyTorch
- Tesseract OCR (Optional, may be used for specific board detection scenarios) ([Installation Guide](https://github.com/UB-Mannheim/tesseract/wiki))
- Stockfish chess engine (included for Windows)

## Installation

1. **Clone the repository**:

    ```bash
    git clone https://github.com/yourusername/ocr-chess.git
    cd ocr-chess
    ```

2. **Create and activate a virtual environment**:

    ```bash
    # Create the environment
    python -m venv .venv

    # Activate the environment
    # Windows (cmd.exe)
    .\.venv\Scripts\activate
    # Windows (PowerShell)
    .\.venv\Scripts\Activate.ps1
    # macOS/Linux (bash/zsh)
    source .venv/bin/activate
    ```

3. **Install dependencies using pip-tools**:

    ```bash
    pip install pip-tools
    pip-sync
    ```

4. **Download piece templates** (if not present):

    ```bash
    python download.py
    ```

## Training the Piece Recognition Model (Optional)

If you want to train the CNN model on your own data:

1. **Generate Square Images**: Use `data_generator.py` to extract individual square images from your chess screenshots or datasets. Place them in the `data/train` and `data/validation` directories, categorized by piece type (e.g., `wP`, `bK`, `empty`). See the `data/` directory structure for examples.
2. **Train the CNN**: Run the training script:

    ```bash
    python train_cnn.py
    ```

    This will create/update the `chess_piece_cnn.pth` model file.

## Usage

Run the main application:

```bash
python main.py
```

The application will detect the chessboard, continuously analyze the position, and display suggested moves via an overlay.

## Technical Details

- **Core Modules**:
  - `chess_engine.py`: Stockfish interface
  - `ocr_utils.py`: Board state analysis & FEN conversion
  - `screen_utils.py`: Screen capture & processing
  - `overlay.py`: Move suggestion display
  - `train_cnn.py`: Script for training the piece recognition model
  - `data_generator.py`: Script for generating training data squares
  - `chess_piece_cnn.pth`: Trained PyTorch model for piece recognition
- **Key Dependencies**:
  - OpenCV
  - PyQt6
  - python-chess
  - Stockfish
  - PyTorch
  - TorchVision
  - pytesseract (Optional)

## Future Enhancements

- Enhance player perspective detection.
- Optimize performance (CPU usage).
- Smoother overlay rendering.
