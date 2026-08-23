"""Regenerate the dependency-layer COPY block in every service Dockerfile.

Each image copies every workspace member's `pyproject.toml` before any source, so
`uv sync` resolves the workspace exactly as it does on a developer machine and the
dependency layer stays cached until a dependency actually changes.

That list has to name every member, which means it goes stale the moment a service is
added, split or renamed -- and a stale list fails at build time with a resolution error
that does not mention the missing file. Generating it removes the class of bug.

Run after adding or renaming a workspace member: `make sync-dockerfiles`.
"""

from pathlib import Path

START = "# --- workspace manifests (generated: make sync-dockerfiles) ---"
END = "# --- end workspace manifests ---"


def manifest_block() -> str:
    """Build the COPY lines for every member's manifest, in a stable order."""
    members = sorted(
        str(p.parent)
        for p in list(Path("libs").glob("*/pyproject.toml"))
        + list(Path("services").glob("*/pyproject.toml"))
    )
    copies = "\n".join(f"COPY {m}/pyproject.toml {m}/" for m in members)
    return f"{START}\n{copies}\n{END}"


def main() -> None:
    """Rewrite the generated block in every service Dockerfile."""
    block = manifest_block()
    for dockerfile in sorted(Path("services").glob("*/Dockerfile*")):
        text = dockerfile.read_text()
        if START not in text or END not in text:
            continue
        head, rest = text.split(START, 1)
        _, tail = rest.split(END, 1)
        dockerfile.write_text(head + block + tail)
        print(f"synced {dockerfile}")


if __name__ == "__main__":
    main()
