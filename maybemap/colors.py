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
    value = kml_color.strip().lstrip("#")
    if len(value) != 8:
        raise ValueError(f"Expected an 8-digit KML aabbggrr color, got {kml_color!r}")
    alpha = int(value[0:2], 16) / 255.0
    blue, green, red = value[2:4], value[4:6], value[6:8]
    return f"#{red}{green}{blue}".lower(), alpha
