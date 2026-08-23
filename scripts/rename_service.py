"""Rename a freshly-copied `services/_template` into a real service.

`make new-service NAME=x` copies the template directory; everything inside it
still calls itself "template". This rewrites the package directory, the project
name, the image names and the imports so the copy is a working workspace member
rather than a second copy of the template.
"""

import re
import sys
from pathlib import Path

SKIP_DIRS = {".venv", "__pycache__", ".mypy_cache", ".pytest_cache", ".ruff_cache"}


def main() -> None:
    """Rewrite `services/<name>` in place from the template's identifiers."""
    if len(sys.argv) != 2:
        raise SystemExit("usage: rename_service.py <service-name>")

    name = sys.argv[1]
    package = name.replace("-", "_")
    root = Path("services") / name
    if not root.is_dir():
        raise SystemExit(f"{root} does not exist")

    # The topic package is named after the service.
    template_pkg = root / "template"
    if template_pkg.is_dir():
        template_pkg.rename(root / package)

    for path in root.rglob("*"):
        if not path.is_file() or set(path.parts) & SKIP_DIRS:
            continue
        if path.suffix not in {".py", ".toml", ".md", ".yaml", ".yml", ".ipynb", ""}:
            continue
        try:
            text = path.read_text()
        except UnicodeDecodeError:
            continue
        updated = re.sub(r"\btemplate\b", package, text)
        updated = updated.replace("ai-residency-template", f"ai-residency-{name}")
        if updated != text:
            path.write_text(updated)

    print(f"services/{name}: package {package}, project ai-residency-{name}")


if __name__ == "__main__":
    main()
