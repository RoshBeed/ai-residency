"""Set every committed PNG's declared resolution to one number, losslessly.

    python deploy/stamp_dpi.py            # report what would change
    python deploy/stamp_dpi.py --write    # rewrite the pHYs chunks

Charts are written at `ai_residency_common.figures.DPI` by the code that draws them, so
they need nothing here. Screenshots and slide exports are the problem: they arrive at
whatever the capturing tool felt like -- 110, 140, 150, 300, 600, or nothing at all --
and a README that mixes them renders figures a reader cannot line up.

This rewrites the `pHYs` chunk only. The pixels, the palette and every other chunk are
untouched, so it is not a re-encode and cannot lose quality; a PNG stores pixels per
metre and has no DPI field of its own, hence the conversion.

`deploy/test_workspace.py::test_every_committed_figure_is_the_portfolio_resolution` is
what makes this stay true.
"""

import argparse
import pathlib
import zlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
DPI = 200
SIGNATURE = b"\x89PNG\r\n\x1a\n"


def _chunk(kind: bytes, body: bytes) -> bytes:
    """One PNG chunk: length, type, body, CRC over type and body."""
    return len(body).to_bytes(4, "big") + kind + body + zlib.crc32(kind + body).to_bytes(4, "big")


def stamp(path: pathlib.Path, dpi: int = DPI) -> bool:
    """Rewrite one PNG's declared resolution.

    Args:
        path: The PNG to stamp.
        dpi: The resolution to declare.

    Returns:
        True if the file's bytes changed.
    """
    raw = path.read_bytes()
    if not raw.startswith(SIGNATURE):
        return False
    per_metre = round(dpi / 0.0254)
    physical = _chunk(b"pHYs", per_metre.to_bytes(4, "big") * 2 + b"\x01")

    out, offset, replaced = bytearray(raw[:8]), 8, False
    while offset < len(raw):
        length = int.from_bytes(raw[offset : offset + 4], "big")
        kind = raw[offset + 4 : offset + 8]
        end = offset + 12 + length
        if kind == b"pHYs":
            out += physical
            replaced = True
        else:
            # pHYs must precede the image data, so a file without one gets it here.
            if kind == b"IDAT" and not replaced:
                out += physical
                replaced = True
            out += raw[offset:end]
        offset = end

    if bytes(out) == raw:
        return False
    path.write_bytes(bytes(out))
    return True


def main() -> None:
    """Stamp every PNG under an `assets/` directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="Rewrite, rather than report.")
    parser.add_argument("--dpi", type=int, default=DPI)
    arguments = parser.parse_args()

    for image in sorted(ROOT.glob("**/assets/**/*.png")):
        if ".venv" in image.parts:
            continue
        if arguments.write:
            if stamp(image, arguments.dpi):
                print(f"stamped {image.relative_to(ROOT)}")
        else:
            print(f"would stamp {image.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
