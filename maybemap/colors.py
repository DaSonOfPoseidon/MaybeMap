# Small color helpers. Plain hex arithmetic, so the build needs no matplotlib.


def _rgb(hex_color: str) -> tuple[int, int, int]:
    value = hex_color.lstrip("#")
    if len(value) != 6:
        raise ValueError(f"Expected a #rrggbb color, got {hex_color!r}")
    return int(value[0:2], 16), int(value[2:4], 16), int(value[4:6], 16)


def darken(hex_color: str, amount: float = 0.35) -> str:
    scale = 1.0 - amount
    return "#" + "".join(f"{round(c * scale):02x}" for c in _rgb(hex_color))


def kml_color_to_hex(kml_color: str) -> tuple[str, float]:
    # KML packs color as aabbggrr, the reverse of the usual rrggbb ordering
    # The result is written into style attributes on the page, so anything but
    # eight hex digits is refused rather than passed through
    value = kml_color.strip().lstrip("#")
    if len(value) != 8 or any(c not in "0123456789abcdefABCDEF" for c in value):
        raise ValueError(f"Expected an 8-digit KML aabbggrr color, got {kml_color!r}")
    alpha = int(value[0:2], 16) / 255.0
    blue, green, red = value[2:4], value[4:6], value[6:8]
    return f"#{red}{green}{blue}".lower(), alpha


# WCAG 2 contrast. AAA asks 7:1 for normal-size text, and nearly every label on the page is
# normal size
AAA_TEXT = 7.0
BLACK = "#000000"
WHITE = "#ffffff"


def luminance(hex_color: str) -> float:
    def channel(c: int) -> float:
        s = c / 255
        return s / 12.92 if s <= 0.04045 else ((s + 0.055) / 1.055) ** 2.4

    r, g, b = (channel(c) for c in _rgb(hex_color))
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def contrast(a: str, b: str) -> float:
    la, lb = sorted((luminance(a), luminance(b)), reverse=True)
    return (la + 0.05) / (lb + 0.05)


def _toward(hex_color: str, target: int, amount: float) -> str:
    return "#" + "".join(f"{round(c + (target - c) * amount):02x}" for c in _rgb(hex_color))


def readable(bg: str, ink: str | None = None, target: float = AAA_TEXT) -> tuple[str, str]:
    # A background and a text color that reach the target. The ink is black or white, whichever
    # reads better unless one is asked for, and the background is only moved (darker under white
    # text, lighter under black) as far as it takes, so the hue mostly survives
    if ink is None:
        ink = WHITE if contrast(WHITE, bg) >= contrast(BLACK, bg) else BLACK
    away = 0 if ink == WHITE else 255
    step = 0
    out = bg.lower()
    while contrast(ink, out) < target:
        step += 1
        out = _toward(bg, away, step / 50)
    return out, ink
