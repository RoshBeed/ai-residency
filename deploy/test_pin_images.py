"""The manifest, tested rather than asserted.

`deploy/images.lock` is what the cluster and compose actually deploy, so the parts
that can go wrong quietly -- a digest that does not propagate, drift that is not
detected, a lock that generates something unparseable -- are worth pinning here.
"""

import pathlib
import subprocess
import sys

DEPLOY = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(DEPLOY))

import pin_images  # noqa: E402

DIGEST = "sha256:" + "a" * 64
OTHER = "sha256:" + "b" * 64


def test_the_lock_is_well_formed() -> None:
    """Every entry is a real digest, not a tag someone pasted in."""
    images = pin_images.load()
    assert images, "the manifest is empty"
    for name, digest in images.items():
        assert digest.startswith("sha256:"), f"{name} is not pinned by digest: {digest}"
        assert len(digest) == 71, f"{name} has a malformed digest: {digest}"


def test_every_service_has_an_entry() -> None:
    """A service the chart deploys but the manifest omits would fall back to a tag."""
    services = {
        p.name
        for p in (DEPLOY.parent / "services").iterdir()
        if p.is_dir() and p.name != "_template"
    }
    assert services <= set(pin_images.load()) | {"base-serve"}


def test_repo_names_match_the_published_layout() -> None:
    """The frontend has no -serve suffix; everything else does."""
    assert pin_images.repo_for("frontend") == "roshb/ai-residency-frontend"
    assert pin_images.repo_for("base-serve") == "roshb/ai-residency-base-serve"
    assert pin_images.repo_for("rlhf-ppo") == "roshb/ai-residency-rlhf-ppo-serve"


def test_values_render_puts_the_digest_under_its_image() -> None:
    text = "backends:\n  word-embeddings:\n    image: roshb/ai-residency-word-embeddings-serve\n"
    out = pin_images.render_values(text, {"word-embeddings": DIGEST})
    assert f"    digest: {DIGEST}\n" in out


def test_values_render_replaces_rather_than_appends() -> None:
    """Running it twice must not stack digest lines under one image."""
    text = "backends:\n  word-embeddings:\n    image: roshb/ai-residency-word-embeddings-serve\n"
    once = pin_images.render_values(text, {"word-embeddings": DIGEST})
    twice = pin_images.render_values(once, {"word-embeddings": OTHER})
    assert twice.count("digest:") == 1
    assert OTHER in twice and DIGEST not in twice


def test_compose_render_rewrites_tag_and_digest_forms() -> None:
    """Both what the file used to hold and what it holds now."""
    repo = "roshb/ai-residency-word-embeddings-serve"
    for existing in (":${IMAGE_TAG:-sha-abc}", "@" + OTHER, ":latest"):
        text = f"  word-embeddings:\n    image: {repo}{existing}\n"
        out = pin_images.render_compose(text, {"word-embeddings": DIGEST})
        assert f"@{DIGEST}" in out, existing
        assert out.count("@sha256:") == 1, existing


def test_the_checked_in_files_match_the_lock() -> None:
    """The guard that makes the manifest trustworthy: no hand edits have drifted."""
    result = subprocess.run(
        [sys.executable, str(DEPLOY / "pin_images.py"), "--check"],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_a_tampered_digest_is_detected() -> None:
    """A silent mismatch between the manifest and what deploys is the failure mode."""
    values = DEPLOY / "helm" / "ai-residency" / "values.yaml"
    original = values.read_text()
    # base-serve is in the manifest but not rendered into values.yaml -- nothing in
    # the chart runs it directly; services build FROM it.
    rendered = [(n, d) for n, d in pin_images.load().items() if f"digest: {d}" in original]
    assert rendered, "no manifest digest reached values.yaml"
    _, digest = rendered[0]
    try:
        values.write_text(original.replace(f"digest: {digest}", f"digest: {DIGEST}", 1))
        result = subprocess.run(
            [sys.executable, str(DEPLOY / "pin_images.py"), "--check"],
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert result.returncode == 1, "drift was not detected"
        assert "Out of sync" in result.stdout
    finally:
        values.write_text(original)
