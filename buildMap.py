# buildMap.py - builds maybe_map.html: one interactive map with zones, city outlines,
# city dots and dispatch drive times as layers that switch on and off independently.

from maybemap import page, payload
from maybemap.branding import load_branding


def main():
    # Read once, first: a bad setting stops the build before any lookup, and the
    # logo and fonts are encoded once for both the payload and the page
    branding = load_branding()
    page.write(payload.build(branding=branding), branding=branding)


if __name__ == "__main__":
    main()
