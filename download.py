import urllib.request
from pathlib import Path

# Wikimedia Commons 128px PNGs for standard chess pieces
WIKI_BASE = "https://upload.wikimedia.org/wikipedia/commons/thumb/"
PIECE_URLS = {
    "wK": f"{WIKI_BASE}4/42/Chess_klt45.svg/128px-Chess_klt45.svg.png",
    "wQ": f"{WIKI_BASE}1/15/Chess_qlt45.svg/128px-Chess_qlt45.svg.png",
    "wR": f"{WIKI_BASE}7/72/Chess_rlt45.svg/128px-Chess_rlt45.svg.png",
    "wB": f"{WIKI_BASE}b/b1/Chess_blt45.svg/128px-Chess_blt45.svg.png",
    "wN": f"{WIKI_BASE}7/70/Chess_nlt45.svg/128px-Chess_nlt45.svg.png",
    "wP": f"{WIKI_BASE}4/45/Chess_plt45.svg/128px-Chess_plt45.svg.png",
    "bK": f"{WIKI_BASE}f/f0/Chess_kdt45.svg/128px-Chess_kdt45.svg.png",
    "bQ": f"{WIKI_BASE}4/47/Chess_qdt45.svg/128px-Chess_qdt45.svg.png",
    "bR": f"{WIKI_BASE}f/ff/Chess_rdt45.svg/128px-Chess_rdt45.svg.png",
    "bB": f"{WIKI_BASE}9/98/Chess_bdt45.svg/128px-Chess_bdt45.svg.png",
    "bN": f"{WIKI_BASE}e/ef/Chess_ndt45.svg/128px-Chess_ndt45.svg.png",
    "bP": f"{WIKI_BASE}c/c7/Chess_pdt45.svg/128px-Chess_pdt45.svg.png",
}

src_folder = Path("python-chess-pieces")
src_folder.mkdir(exist_ok=True)

USER_AGENT = "OCRChessBot/1.0 (contact: your_email@example.com)"

if __name__ == "__main__":
    for piece, url in PIECE_URLS.items():
        if not url.startswith("http"):
            msg = f"Refusing to open non-http URL: {url}"
            raise ValueError(msg)
        filename = f"{piece}.png"
        dest_path = src_folder / filename
        if not dest_path.exists():
            print(f"Downloading {filename} from Wikimedia Commons...")
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with (
                urllib.request.urlopen(req) as response,
                dest_path.open("wb") as out_file,
            ):
                out_file.write(response.read())
        else:
            print(f"{filename} already exists, skipping download.")
    print("All piece PNGs are downloaded in 'python-chess-pieces'.")
