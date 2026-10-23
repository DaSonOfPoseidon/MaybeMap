# buildMap.py - builds maybe_map.html: one interactive map with zones, city outlines,
# city dots and dispatch drive times as layers that switch on and off independently.

from maybemap import page, payload


def main():
    page.write(payload.build())


if __name__ == "__main__":
    main()
