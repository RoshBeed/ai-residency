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

    python deploy/pin_images.py                       # regenerate values.yaml and compose
    python deploy/pin_images.py --check               # fail if they have drifted
    python deploy/pin_images.py --update a b --sha X  # resolve digests for a and b
"""

import argparse
import json
import pathlib
import re
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent
LOCK = ROOT / "images.lock"
VALUES = ROOT / "helm" / "ai-residency" / "values.yaml"
COMPOSE = ROOT / "compose" / "docker-compose.yml"


def load() -> dict[str, str]:
    """The manifest: service name to `sha256:...`."""
    data = json.loads(LOCK.read_text())
    return dict(data["images"])


def repo_for(name: str) -> str:
    """The Docker Hub repository for a service name."""
    if name in {"base-serve", "frontend"}:
        return f"roshb/ai-residency-{name}"
    return f"roshb/ai-residency-{name}-serve"


def resolve(name: str, sha: str) -> str:
    """The digest the registry currently has for `name` at `sha-<commit>`.

    Only images CI actually rebuilt have that tag. Everything else keeps the digest
    already in the lock, which is the point: an unchanged image needs no new tag, and
    the manifest is what carries "the set", so nothing has to be minted for it.
    """
    ref = f"{repo_for(name)}:sha-{sha}"
    out = subprocess.run(
        ["docker", "buildx", "imagetools", "inspect", ref, "--format", "{{.Manifest.Digest}}"],
        capture_output=True,
        text=True,
        timeout=180,
    )
    digest = out.stdout.strip()
    if not digest.startswith("sha256:"):
        raise SystemExit(f"could not resolve {ref}\n{out.stderr.strip()[:400]}")
    return digest


def update(names: list[str], sha: str) -> bool:
    """Point `names` at whatever the registry has for them at `sha`. Returns True if changed."""
    data = json.loads(LOCK.read_text())
    images = data["images"]
    changed = False
    for name in names:
        digest = resolve(name, sha)
        if images.get(name) != digest:
            print(f"{name:<30} {str(images.get(name))[:19]} -> {digest[:19]}")
            images[name] = digest
            changed = True
        else:
            print(f"{name:<30} unchanged")
    if changed:
        LOCK.write_text(json.dumps(data, indent=2) + "\n")
    return changed


def render_values(text: str, images: dict[str, str]) -> str:
    """Set `digest:` under each backend and the frontend."""
    for name, digest in images.items():
        if name == "base-serve":
            continue
        # `digest:` sits directly under the service's `image:` line, at the same indent.
        pattern = re.compile(
            rf"(^(?P<indent>\s+)image: roshb/ai-residency-{re.escape(name)}(?:-serve)?\n)"
            rf"(?P<existing>(?P=indent)digest: .*\n)?",
            re.M,
        )
        if not pattern.search(text):
            continue
        # `digest=digest` binds this iteration's value: a bare closure over the loop
        # variable is only correct because re.sub happens to run it immediately.
        text = pattern.sub(
            lambda m, digest=digest: f"{m.group(1)}{m.group('indent')}digest: {digest}\n",
            text,
        )
    return text


def render_compose(text: str, images: dict[str, str]) -> str:
    """Rewrite each `image:` to `repo@sha256:...`, dropping the ${IMAGE_TAG} default."""
    for name, digest in images.items():
        if name == "base-serve":
            continue
        repo = f"roshb/ai-residency-{name}"
        if name != "frontend":
            repo += "-serve"
        pattern = (
            rf"(^\s+image: {re.escape(repo)})"
            rf"(?::\$\{{IMAGE_TAG:-[^}}]*\}}|@sha256:[0-9a-f]+|:[^\s@]+)"
        )
        text = re.sub(
            pattern,
            lambda m, digest=digest: f"{m.group(1)}@{digest}",
            text,
            flags=re.M,
        )
    return text


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="fail on drift instead of writing")
    parser.add_argument("--update", nargs="*", metavar="NAME", help="images to re-resolve")
    parser.add_argument("--sha", help="commit whose sha- tag holds the rebuilt images")
    args = parser.parse_args()

    if args.update is not None:
        if not args.sha:
            parser.error("--update needs --sha")
        if not args.update:
            print("nothing was rebuilt; the manifest already describes this commit")
        else:
            update(args.update, args.sha)

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
            print(f"updated {path.relative_to(ROOT.parent)}")

    if drifted:
        print("Out of sync with deploy/images.lock: " + ", ".join(str(p) for p in drifted))
        print("Run: python deploy/pin_images.py")
        return 1
    if args.check:
        print("compose and the chart match deploy/images.lock")
    return 0


if __name__ == "__main__":
    sys.exit(main())
