"""The manifest, tested rather than asserted.

`deploy/images.yaml` is what the cluster actually deploys, so the parts that can go
wrong quietly are worth pinning here.

There is much less to test than there was. The script used to rewrite `values.yaml` and
`docker-compose.yml` with regular expressions, and most of this file existed to prove
those substitutions replaced rather than appended, matched both the old and new forms,
and were detected when they drifted. Helm reads the manifest directly as a values
overlay now and Compose substitutes from an env file, so the only generated artifact
left is that env file -- and the only propagation bug still possible is it being stale.
"""

import pathlib
import subprocess
import sys

import yaml

DEPLOY = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(DEPLOY))

import pin_images  # noqa: E402

DIGEST = "sha256:" + "a" * 64


def test_the_manifest_is_well_formed() -> None:
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
    assert services <= set(pin_images.load()) | {pin_images.BASE}


def test_repo_names_match_the_published_layout() -> None:
    """The frontend has no -serve suffix; everything else does."""
    assert pin_images.repo_for("frontend") == "roshb/ai-residency-frontend"
    assert pin_images.repo_for(pin_images.BASE) == "roshb/ai-residency-base-serve"
    assert pin_images.repo_for("rlhf-ppo") == "roshb/ai-residency-rlhf-ppo-serve"


def test_the_manifest_is_a_valid_helm_values_overlay() -> None:
    """It is passed to `helm -f` directly, so its shape has to match the chart's."""
    manifest = yaml.safe_load(pin_images.MANIFEST.read_text())
    values = yaml.safe_load((DEPLOY / "helm/ai-residency/values.yaml").read_text())
    assert set(manifest) <= set(values), "the overlay names keys the chart does not have"
    assert set(manifest["backends"]) == set(values["backends"]), (
        "the overlay and the chart disagree about which backends exist"
    )
    for name, spec in manifest["backends"].items():
        assert set(spec) == {"digest"}, f"{name} carries more than a digest: {sorted(spec)}"


def test_values_yaml_holds_no_digests() -> None:
    """The whole point: one place names the images, so there is nothing to keep in step.

    A digest here would still render -- Helm merges rather than replaces -- and would
    silently win for any service the overlay happens not to name.
    """
    text = (DEPLOY / "helm/ai-residency/values.yaml").read_text()
    assert "sha256:" not in text, "values.yaml has digests again; they belong in images.yaml"


def test_the_env_file_covers_every_deployed_image() -> None:
    """A service missing from it resolves to `:latest` under compose, silently."""
    images = {n: d for n, d in pin_images.load().items() if n != pin_images.BASE}
    rendered = pin_images.render_env(images)
    for name, digest in images.items():
        assert f"{pin_images.env_var_for(name)}=@{digest}" in rendered, name


def test_env_var_names_match_the_compose_file() -> None:
    """The two are written by different hands; a mismatch falls back to `:latest`."""
    compose = (DEPLOY / "compose/docker-compose.yml").read_text()
    for name in pin_images.load():
        if name == pin_images.BASE:
            continue
        var = pin_images.env_var_for(name)
        assert f"${{{var}:-:latest}}" in compose, f"docker-compose.yml never uses {var}"


def test_the_checked_in_env_file_matches_the_manifest() -> None:
    """The one guard left: the only generated file, and the only thing that can drift."""
    result = subprocess.run(
        [sys.executable, str(DEPLOY / "pin_images.py"), "--check"],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_a_stale_env_file_is_detected() -> None:
    """Drift here is silent otherwise: compose would deploy the previous digest."""
    env = pin_images.COMPOSE_ENV
    original = env.read_text()
    try:
        env.write_text(original.replace("=@sha256:", "=@sha256:0", 1))
        result = subprocess.run(
            [sys.executable, str(DEPLOY / "pin_images.py"), "--check"],
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert result.returncode == 1, "a stale env file passed --check"
        assert "out of date" in result.stdout
    finally:
        env.write_text(original)
