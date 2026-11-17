# Small color helpers. Plain hex arithmetic, so the build needs no matplotlib.

import math


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


def _oklch(hex_color: str) -> tuple[float, float, float]:
    # sRGB hex to OKLCH (lightness, chroma, hue in radians)
    lin = [
        s / 12.92 if s <= 0.04045 else ((s + 0.055) / 1.055) ** 2.4
        for s in (c / 255 for c in _rgb(hex_color))
    ]
    r, g, b = lin
    lms = [
        (0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b) ** (1 / 3),
        (0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b) ** (1 / 3),
        (0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b) ** (1 / 3),
    ]
    lk = 0.2104542553 * lms[0] + 0.7936177850 * lms[1] - 0.0040720468 * lms[2]
    a = 1.9779984951 * lms[0] - 2.4285922050 * lms[1] + 0.4505937099 * lms[2]
    bb = 0.0259040371 * lms[0] + 0.7827717662 * lms[1] - 0.8086757660 * lms[2]
    return lk, math.hypot(a, bb), math.atan2(bb, a)


def _from_oklch(lk: float, c: float, h: float) -> str:
    a, b = c * math.cos(h), c * math.sin(h)
    l_ = (lk + 0.3963377774 * a + 0.2158037573 * b) ** 3
    m_ = (lk - 0.1055613458 * a - 0.0638541728 * b) ** 3
    s_ = (lk - 0.0894841775 * a - 1.2914855480 * b) ** 3
    lin = (
        4.0767416621 * l_ - 3.3077115913 * m_ + 0.2309699292 * s_,
        -1.2684380046 * l_ + 2.6097574011 * m_ - 0.3413193965 * s_,
        -0.0041960863 * l_ - 0.7034186147 * m_ + 1.7076147010 * s_,
    )
    # Clamped to sRGB the way a browser maps an out-of-gamut color
    out = [max(0.0, min(1.0, x)) for x in lin]
    enc = [12.92 * x if x <= 0.0031308 else 1.055 * x ** (1 / 2.4) - 0.055 for x in out]
    return "#" + "".join(f"{round(x * 255):02x}" for x in enc)


def lift_lightness(hex_color: str, floor: float) -> str:
    # The color with its OKLCH lightness raised to the floor, hue and chroma kept. This is what
    # the CSS relative color oklch(from c max(l, floor) c h) gives, computed here so older
    # browsers that cannot parse relative colors still get a plain hex value
    lk, c, h = _oklch(hex_color)
    if lk >= floor:
        return hex_color.lower()
    return _from_oklch(floor, c, h)
