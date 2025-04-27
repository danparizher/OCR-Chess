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
- **Automatic Board & Piece Detection**: Identifies chessboard regions and recognizes pieces using template matching.
- **FEN Generation & Analysis**: Converts positions to FEN and uses the Stockfish engine for best move suggestions.
- **Visual Overlay**: Displays suggested moves directly on the screen.
- **Perspective Handling**: Works for both white and black perspectives.

## Requirements

- Python 3.13
- Tesseract OCR ([Installation Guide](https://github.com/UB-Mannheim/tesseract/wiki))
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

## Usage

Run the main application:

```bash
python main.py
```

The application will detect the chessboard, continuously analyze the position, and display suggested moves via an overlay.

## Technical Details

- **Core Modules**:
  - `chess_engine.py`: Stockfish interface
  - `ocr_utils.py`: Piece recognition & FEN conversion
  - `screen_utils.py`: Screen capture & processing
  - `overlay.py`: Move suggestion display
- **Key Dependencies**:
  - OpenCV
  - PyQt6
  - python-chess
  - Stockfish
  - pytesseract (optional)

## Future Enhancements

- Enhance player perspective detection.
- Optimize performance (CPU usage).
- Smoother overlay rendering.
