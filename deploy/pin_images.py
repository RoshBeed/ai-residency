#!/usr/bin/env python3
"""Propagate `deploy/images.lock` into the chart's values and the compose file.

The lock is the manifest: one file naming, by immutable digest, the exact image
every service runs. It exists because a tag cannot honestly carry that meaning. A
tag describes *one image's* content; `deploy/` needed something describing *the
whole set*, and overloading a single `image.tag` across eleven images only works
while every image changes on every commit. The moment one service changes and eight
do not, the tag either lies or has to be minted for images that did not earn it.

Digests also cannot be moved, which a tag can. `git log deploy/images.lock` is the
deployment history and `git revert` is the rollback.

    python deploy/pin_images.py           # regenerate values.yaml and compose
    python deploy/pin_images.py --check    # fail if they have drifted
"""

import argparse
import json
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent
LOCK = ROOT / "images.lock"
VALUES = ROOT / "helm" / "ai-residency" / "values.yaml"
COMPOSE = ROOT / "compose" / "docker-compose.yml"


def load() -> dict[str, str]:
    """The manifest: service name to `sha256:...`."""
    data = json.loads(LOCK.read_text())
    return dict(data["images"])


def render_values(text: str, images: dict[str, str]) -> str:
    """Set `digest:` under each backend and the frontend."""
    for name, digest in images.items():
        if name == "base-serve":
            continue
        # `digest:` sits directly under the service's `image:` line, at the same indent.
        pattern = re.compile(
            r"(^(?P<indent>\s+)image: roshb/ai-residency-%s(?:-serve)?\n)"
            r"(?P<existing>(?P=indent)digest: .*\n)?" % re.escape(name),
            re.M,
        )
        if not pattern.search(text):
            continue
        text = pattern.sub(
            lambda m: "%s%sdigest: %s\n" % (m.group(1), m.group("indent"), digest), text
        )
    return text


def render_compose(text: str, images: dict[str, str]) -> str:
    """Rewrite each `image:` to `repo@sha256:...`, dropping the ${IMAGE_TAG} default."""
    for name, digest in images.items():
        if name == "base-serve":
            continue
        repo = "roshb/ai-residency-%s" % name
        if name != "frontend":
            repo += "-serve"
        text = re.sub(
            r"(^\s+image: %s)(?::\$\{IMAGE_TAG:-[^}]*\}|@sha256:[0-9a-f]+|:[^\s@]+)"
            % re.escape(repo),
            lambda m: "%s@%s" % (m.group(1), digest),
            text,
            flags=re.M,
        )
    return text


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="fail on drift instead of writing")
    args = parser.parse_args()

    images = load()
    drifted = []
    for path, render in ((VALUES, render_values), (COMPOSE, render_compose)):
        before = path.read_text()
        after = render(before, images)
        if before == after:
            continue
        if args.check:
            drifted.append(path.relative_to(ROOT.parent))
        else:
            path.write_text(after)
            print("updated %s" % path.relative_to(ROOT.parent))

    if drifted:
        print("Out of sync with deploy/images.lock: %s" % ", ".join(str(p) for p in drifted))
        print("Run: python deploy/pin_images.py")
        return 1
    if args.check:
        print("compose and the chart match deploy/images.lock")
    return 0


if __name__ == "__main__":
    sys.exit(main())
