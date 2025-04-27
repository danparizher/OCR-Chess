from __future__ import annotations

import ctypes
import sys

from PyQt6 import QtCore, QtGui, QtWidgets


class ChessOverlay(QtWidgets.QWidget):
    def __init__(
        self,
        region: tuple[int, int, int, int],
        move: str | None = None,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        print(f"[Overlay Debug] Region: {region}")
        screen = QtWidgets.QApplication.primaryScreen()
        if screen:
            print(f"[Overlay Debug] Screen geometry: {screen.geometry()}")
        self.setWindowFlags(
            QtCore.Qt.WindowType.FramelessWindowHint
            | QtCore.Qt.WindowType.WindowStaysOnTopHint
            | QtCore.Qt.WindowType.Tool,
        )
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.setGeometry(region[0], region[1], region[2], region[3])
        self.move_str = move
        self.square_size = (region[2] / 8, region[3] / 8)
        self.show()

        # --- Make window click-through on Windows ---
        if sys.platform.startswith("win"):
            hwnd = int(self.winId())
            GWL_EXSTYLE = -20
            WS_EX_LAYERED = 0x80000
            WS_EX_TRANSPARENT = 0x20
            user32 = ctypes.windll.user32
            style = user32.GetWindowLongW(hwnd, GWL_EXSTYLE)
            style |= WS_EX_LAYERED | WS_EX_TRANSPARENT
            user32.SetWindowLongW(hwnd, GWL_EXSTYLE, style)

    def move_to_indices(self, move: str) -> tuple[tuple[int, int], tuple[int, int]]:
        # e.g., 'e2e4' -> ((6, 4), (4, 4)) (row, col)
        def algebraic_to_index(square: str) -> tuple[int, int]:
            col = ord(square[0]) - ord("a")
            row = 8 - int(square[1])
            return (row, col)

        from_sq = move[:2]
        to_sq = move[2:4]
        return algebraic_to_index(from_sq), algebraic_to_index(to_sq)

    def paintEvent(self, event: QtGui.QPaintEvent | None) -> None:
        _ = event
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        # Draw debug border
        pen = QtGui.QPen(QtGui.QColor(255, 255, 0, 200))
        pen.setWidth(4)
        painter.setPen(pen)
        painter.drawRect(0, 0, self.width() - 1, self.height() - 1)
        # Draw move highlights as borders only if move_str is valid
        if self.move_str and isinstance(self.move_str, str) and len(self.move_str) == 4:
            from_idx, to_idx = self.move_to_indices(self.move_str)
            for idx, color in zip(
                [from_idx, to_idx],
                [QtGui.QColor(0, 255, 0, 200), QtGui.QColor(255, 0, 0, 200)],
            ):
                row, col = idx
                x = col * self.square_size[0]
                y = row * self.square_size[1]
                rect = QtCore.QRectF(x, y, self.square_size[0], self.square_size[1])
                border_pen = QtGui.QPen(color)
                border_pen.setWidth(6)
                painter.setPen(border_pen)
                painter.drawRect(rect)

    def update_move(self, move: str | None) -> None:
        self.move_str = move
        self.update()


def show_overlay(
    region: tuple[int, int, int, int],
    move: str | None = None,
) -> tuple[QtWidgets.QApplication, ChessOverlay]:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    overlay = ChessOverlay(region, move)
    overlay.show()
    # Ensure app is of type QApplication, not QCoreApplication
    if not isinstance(app, QtWidgets.QApplication):
        msg = "QApplication instance could not be created."
        raise TypeError(msg)
    return app, overlay


# Example usage (uncomment to test standalone):
# if __name__ == "__main__":
#     show_overlay((100, 100, 400, 400), "e2e4")
