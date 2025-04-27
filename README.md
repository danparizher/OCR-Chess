# OCR Chess - Computer Vision Chess Assistant

**Disclaimer**: This project is for educational purposes only. It does not include mouse automation or any functionality to interact with chess interfaces, as such features could be misused for cheating in live matches. Use responsibly.

## Features

- **Automatic Chessboard Detection**: Finds chessboard regions on screen
- **Piece Recognition**: Uses template matching to identify chess pieces
- **FEN Generation**: Converts recognized positions to FEN notation
- **Move Suggestion**: Integrates with Stockfish engine for best move analysis
- **Visual Overlay**: Displays suggested moves directly on screen
- **Perspective Handling**: Works for both white and black perspectives

## What Makes This Stand Out?

Unlike many chess assistants that rely on browser automation (e.g., Playwright or Selenium) or specific website integrations, this project uses **pure computer vision** to analyze chess positions from **any screen source**. This makes it uniquely versatile:

- **Works with Chess960 (Fischer Random Chess)**: Detects and analyzes non-standard starting positions.
- **Compatible with Chess Variants**: Handles any variant where the board follows traditional 8x8 geometry (e.g., Horde Chess, Three-Check).
- **No Website Dependencies**: Works with physical boards (via camera) or any digital chess interface, including offline games.
- **Flexible Input**: Processes screenshots, live screen captures, or even images of physical boards.

## Requirements

- Python 3.13
- Tesseract OCR (for text recognition)
- Stockfish chess engine (included for Windows)

## Installation

1. Clone this repository:

   ```bash
   git clone https://github.com/yourusername/ocr-chess.git
   cd ocr-chess
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Download chess piece templates (automatically done by `download.py`):

   ```bash
   python download.py
   ```

4. Install Tesseract OCR from [https://github.com/UB-Mannheim/tesseract/wiki](https://github.com/UB-Mannheim/tesseract/wiki)

## Usage

Run the main application:

```bash
python main.py
```

The application will:

1. Detect the chessboard region on your screen
2. Continuously analyze the board position
3. Display suggested moves via an overlay

## Configuration

Edit `config.py` to adjust:

- Board size (default 8x8)
- Tesseract executable path
- Highlight colors and thickness

## Technical Details

### Key Components

- **Chess Engine (`chess_engine.py`)**: Interfaces with Stockfish
- **OCR Utilities (`ocr_utils.py`)**: Handles piece recognition and FEN conversion
- **Screen Capture (`screen_utils.py`)**: Captures and processes screen regions
- **Visual Overlay (`overlay.py`)**: Displays move suggestions

### Dependencies

- OpenCV: Computer vision processing
- PyQt6: Overlay GUI
- python-chess: Chess logic and FEN handling
- Stockfish: Chess engine analysis
- pytesseract: Optional text recognition

## License

MIT License - See [LICENSE](LICENSE) for details

## Future Enhancements

- **Improved OCR Recognition**: Enhance accuracy for smaller board sizes where current recognition may fail.
- **Player Perspective Detection**: Better detection of whether the user is playing as white or black.
- **Performance Optimizations**: Reduce CPU consumption for smoother operation, especially during continuous analysis.
- **Smoother Overlay**: Optimize the overlay rendering to reduce lag and improve visual fluidity.
