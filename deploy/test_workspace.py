"""The workspace, and the two places that have to keep agreeing with it.

CI selects what to check from a list of path filters, and the build tooling reads a
training entrypoint out of each service's manifest. Both are written down rather than
derived from the directory tree -- readable, and the diff for adding a service shows
what will start running -- but a written-down list goes stale silently. A member with
no filter is simply never selected: nothing lints it, type-checks it or tests it, and
CI stays green while doing none of it.

That is the failure these tests exist to make loud. They live in `deploy/` because
that is the one suite the lint job runs outside the per-member matrix; nothing here is
specific to deploying.
"""

import pathlib
import re
import subprocess
import tomllib
from typing import Any

import yaml

ROOT = pathlib.Path(__file__).resolve().parent.parent
WORKFLOW = ROOT / ".github" / "workflows" / "ci.yml"


def filters() -> dict[str, Any]:
    """The `filters:` block of the paths-filter step, parsed as the action parses it."""
    workflow = yaml.safe_load(WORKFLOW.read_text())
    for step in workflow["jobs"]["changes"]["steps"]:
        if step.get("id") == "filter":
            parsed = yaml.safe_load(step["with"]["filters"])
            assert isinstance(parsed, dict)
            return parsed
    raise AssertionError("no step with id 'filter' in the changes job")


def patterns_for(parsed: dict[str, Any], name: str) -> list[str]:
    """One filter's patterns, flattened the way the action flattens them.

    A YAML alias to a list nested inside a list parses as `[[...], 'services/x/**']`,
    which `parseFilterItemYaml` in the action flattens by recursing into every array.
    Comparing against the unflattened shape would assert something the action never
    sees.
    """

    def flatten(item: Any) -> list[str]:
        if isinstance(item, list):
            return [pattern for element in item for pattern in flatten(element)]
        return [item]

    return flatten(parsed[name])


def members() -> set[str]:
    """Every workspace member, by basename -- which is what a filter is named."""
    return {
        path.name
        for parent in ("libs", "services")
        for path in (ROOT / parent).iterdir()
        if (path / "pyproject.toml").exists()
    }


def test_every_member_has_a_filter() -> None:
    """A member with no filter is never selected, and CI passes without running it."""
    missing = members() - set(filters())
    assert not missing, f"no CI path filter for: {', '.join(sorted(missing))}"


def test_no_filter_names_a_member_that_is_gone() -> None:
    """A stale filter puts a nonexistent path in the check matrix, which then fails."""
    # `shared` is the anchor every member's filter aliases, not a member itself.
    extra = set(filters()) - members() - {"shared"}
    assert not extra, f"CI filters name members that do not exist: {', '.join(sorted(extra))}"


def test_each_member_filter_includes_the_shared_inputs() -> None:
    """A change to libs/ or the lockfile has to select everything, or CI misses breakage."""
    parsed = filters()
    shared = patterns_for(parsed, "shared")
    for name in members():
        patterns = patterns_for(parsed, name)
        missing = set(shared) - set(patterns)
        assert not missing, f"{name} does not inherit the shared inputs: {sorted(missing)}"


def test_services_are_filtered_on_their_own_directory() -> None:
    """Otherwise a service is only ever built when something shared changes."""
    parsed = filters()
    for path in (ROOT / "services").iterdir():
        if not (path / "pyproject.toml").exists():
            continue
        assert f"services/{path.name}/**" in patterns_for(parsed, path.name), (
            f"{path.name}'s filter does not watch its own directory"
        )


def publishable() -> list[pathlib.Path]:
    """Services that get built into images. `_template` is checked but never published."""
    return sorted(
        path
        for path in (ROOT / "services").iterdir()
        if (path / "pyproject.toml").exists() and path.name != "frontend"
    )


def test_every_service_declares_its_training_entrypoint() -> None:
    """One root Dockerfile.train builds them all, and this is the only value it needs.

    Missing, the build fails on the ARG guard rather than producing an image that
    starts and immediately cannot import what it was told to run.
    """
    for path in publishable():
        manifest = tomllib.loads((path / "pyproject.toml").read_text())
        module = manifest.get("tool", {}).get("ai-residency", {}).get("train-module")
        assert module, f"{path.name}/pyproject.toml declares no [tool.ai-residency] train-module"
        assert not module.endswith(".py"), f"{path.name}: {module} is a path, not a module"


def test_the_declared_training_module_exists() -> None:
    """A renamed trainer that nothing catches until a GPU box has been rented."""
    for path in publishable():
        manifest = tomllib.loads((path / "pyproject.toml").read_text())
        module = manifest["tool"]["ai-residency"]["train-module"]
        parts = module.split(".")
        target = path.joinpath(*parts)
        assert target.with_suffix(".py").exists() or (target / "__init__.py").exists(), (
            f"{path.name} declares train-module {module}, which does not exist"
        )


def test_the_root_dockerfiles_are_the_only_ones() -> None:
    """Ten near-identical copies each is what the parameterised pair replaced."""
    strays = [
        path.relative_to(ROOT)
        for path in (ROOT / "services").glob("*/Dockerfile.*")
        # The frontend builds FROM python directly and runs Streamlit, not a
        # FastAPI app -- two branches wrapped around two shared lines.
        if path.parent.name != "frontend"
    ]
    assert not strays, f"per-service Dockerfiles are back: {strays}"
    for name in ("Dockerfile.base", "Dockerfile.serve", "Dockerfile.train"):
        assert (ROOT / name).exists(), f"{name} is missing from the repository root"


def compose_services() -> dict[str, Any]:
    """The compose stack's services, as YAML rather than as Compose resolves them.

    Reading the file directly keeps this a unit test: `docker compose config` would be
    the more faithful source, but it needs a Docker daemon, and this assertion is about
    two lists in one file agreeing with each other.
    """
    document = yaml.safe_load((ROOT / "deploy/compose/docker-compose.yml").read_text())
    services: dict[str, Any] = document["services"]
    return services


def compose_backends() -> set[str]:
    """The services the frontend has to be able to reach: those exposing port 8000."""
    return {
        name
        for name, spec in compose_services().items()
        if any(str(port).endswith(":8000") for port in spec.get("ports", []))
    }


def test_the_compose_frontend_can_reach_every_backend() -> None:
    """A hand-written list in a file that also defines what it has to match.

    The chart generates these from `.Values.backends`' keys, so it cannot go stale.
    Compose has no loops, so its `<SERVICE>_API_URL` entries are written out -- and the
    failure is quiet. A backend with no entry falls back to `http://localhost:800N`,
    which inside the frontend container is the frontend itself, so its page reports
    "this project's backend isn't ready" and reads exactly like a backend that is down.
    """
    frontend = compose_services()["frontend"].get("environment") or {}
    declared = {key for key in frontend if key.endswith("_API_URL")}
    expected = {name.upper().replace("-", "_") + "_API_URL" for name in compose_backends()}
    assert expected - declared == set(), (
        f"the compose frontend cannot reach: {sorted(expected - declared)}"
    )
    assert declared - expected == set(), (
        f"the compose frontend names backends that do not exist: {sorted(declared - expected)}"
    )


def test_every_compose_backend_url_points_at_that_service() -> None:
    """Compose resolves a service name as DNS, so the host must be the service name."""
    frontend = compose_services()["frontend"].get("environment") or {}
    for name in compose_backends():
        url = frontend[name.upper().replace("-", "_") + "_API_URL"]
        assert url == f"http://{name}:8000", f"{name} is wired to {url}"


def chart_values() -> dict[str, Any]:
    return dict(yaml.safe_load((ROOT / "deploy/helm/ai-residency/values.yaml").read_text()))


def test_the_chart_names_no_image_it_can_derive() -> None:
    """The convention is `<registry>/<chart>-<service>-serve`, derived from the range key.

    Written out per backend it went stale the quiet way: rename a key and the Deployment,
    the Service and the frontend's env var all follow, while `image:` goes on naming the
    old repository. An explicit `image:` is still allowed for a service that genuinely
    breaks the convention -- this fails only on one that does not.
    """
    values = chart_values()
    registry = values["image"]["registry"]
    for name, spec in values["backends"].items():
        for key, suffix in (("image", "-serve"), ("trainImage", "-train")):
            named = spec.get(key)
            if named is None:
                continue
            assert named != f"{registry}/ai-residency-{name}{suffix}", (
                f"{name}.{key} spells out what the chart derives; delete the line"
            )


def test_the_hf_cache_mounts_where_the_image_writes() -> None:
    """Two copies of one path is how compose ended up mounting the cache into a void.

    `Dockerfile.base` sets HF_HOME once, saying in its own comment that it lives there so
    the path the chart mounts and the path the library writes cannot drift apart. This is
    the assertion that makes that true.
    """
    base = (ROOT / "Dockerfile.base").read_text()
    match = re.search(r"^ENV HF_HOME=(\S+)$", base, re.M)
    assert match, "Dockerfile.base no longer sets HF_HOME"
    assert chart_values()["hfCache"]["mountPath"] == match.group(1)


def test_the_chart_does_not_restate_hf_home() -> None:
    """The image supplies it; a second copy can only ever be wrong later."""
    for name, spec in chart_values()["backends"].items():
        assert "HF_HOME" not in (spec.get("env") or {}), (
            f"{name} sets HF_HOME, which Dockerfile.base already sets"
        )


def test_compose_does_not_restate_a_working_directory() -> None:
    """Both Dockerfiles end `WORKDIR /app/services/${SERVICE}`, and both CMDs are
    resolved relative to it -- `python -m serving.api` for serving, `python -m
    <train-module>` for training. A `working_dir` in compose silently replaces that.

    `multimodal-captioning` carried `working_dir: /app` on both its serving and its
    training service, so the published image exited immediately on
    `ModuleNotFoundError: No module named 'serving'` while the image itself was
    correct. Nothing caught it: the image builds, CI's smoke test imports the app
    rather than starting the container, and the chart never restated the path.
    """
    compose = yaml.safe_load((ROOT / "deploy/compose/docker-compose.yml").read_text())
    for name, spec in compose["services"].items():
        assert "working_dir" not in spec, (
            f"{name} sets working_dir, which its Dockerfile already sets correctly"
        )


# --- the write-up ---------------------------------------------------------
#
# A service README is the write-up *and* the GitHub landing page, and the frontend
# links to it and reuses its figures. That makes three ways for it to rot quietly: an
# image reference that no longer resolves renders as a broken icon on GitHub, a
# frontend diagram path that no longer exists raises at page render, and a `docs_path`
# pointing at a moved README gives a 404 from the UI. None of them fail a build.


def service_readmes() -> list[pathlib.Path]:
    """Every service README, which is where the write-ups live."""
    return sorted((ROOT / "services").glob("*/README.md"))


def test_every_readme_image_resolves() -> None:
    """A markdown image whose target is gone renders as a broken icon on GitHub."""
    missing = []
    for readme in service_readmes():
        for target in re.findall(r"!\[[^\]]*\]\(([^)]+)\)", readme.read_text()):
            if target.startswith(("http://", "https://")):
                continue
            if not (readme.parent / target).exists():
                missing.append(f"{readme.relative_to(ROOT)} -> {target}")
    assert not missing, "README images that do not resolve: " + ", ".join(missing)


def test_every_readme_image_has_alt_text() -> None:
    """Alt text is the figure's description for anyone who cannot see it."""
    bare = []
    for readme in service_readmes():
        for alt, target in re.findall(r"!\[([^\]]*)\]\(([^)]+)\)", readme.read_text()):
            if not alt.strip():
                bare.append(f"{readme.relative_to(ROOT)} -> {target}")
    assert not bare, "README images with no alt text: " + ", ".join(bare)


def frontend_projects() -> dict[str, Any]:
    """The frontend's project table, imported without starting Streamlit."""
    import importlib.util

    path = ROOT / "services" / "frontend" / "projects.py"
    spec = importlib.util.spec_from_file_location("_frontend_projects", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    projects: dict[str, Any] = module.PROJECTS
    return projects


def test_every_frontend_diagram_resolves() -> None:
    """A missing diagram path raises when the page renders, not when it is built."""
    assets = ROOT / "services" / "frontend"
    missing = [
        f"{name} -> {path}"
        for name, project in frontend_projects().items()
        for path, _ in project.get("diagrams", [])
        if not (assets / path).exists()
    ]
    assert not missing, "frontend diagrams that do not resolve: " + ", ".join(missing)


def test_every_frontend_docs_path_resolves() -> None:
    """`docs_path` becomes a github.com/blob/main/... link; a stale one 404s."""
    missing = [
        f"{name} -> {project['docs_path']}"
        for name, project in frontend_projects().items()
        if "docs_path" in project and not (ROOT / project["docs_path"]).exists()
    ]
    assert not missing, "frontend docs_path values that do not resolve: " + ", ".join(missing)


def png_dpi(path: pathlib.Path) -> int | None:
    """The horizontal resolution a PNG declares, in dots per inch, or None.

    Args:
        path: The PNG to read.

    Returns:
        The rounded DPI, or None if the file declares no physical size.

    Reads the `pHYs` chunk directly rather than through Pillow, which is not a
    dependency of this suite and would be a 3MB install to read eight bytes. The chunk
    stores pixels per metre; PNG has no DPI field of its own.
    """
    with path.open("rb") as handle:
        if handle.read(8) != b"\x89PNG\r\n\x1a\n":
            return None
        while chunk := handle.read(8):
            length = int.from_bytes(chunk[:4], "big")
            if chunk[4:] == b"pHYs":
                body = handle.read(length)
                if body[8] != 1:  # unit 1 is metres; 0 means "aspect ratio only"
                    return None
                return round(int.from_bytes(body[:4], "big") * 0.0254)
            if chunk[4:] == b"IDAT":
                return None
            handle.seek(length + 4, 1)
    return None


def test_every_committed_figure_is_the_portfolio_resolution() -> None:
    """One DPI across every write-up, so two figures are the same size on a page.

    Nine services drew charts with their own `dpi=` and the committed assets came out at
    110, 140, 150, 200, 300 and 600, which a reader sees as figures that will not line
    up. `ai_residency_common.figures.DPI` is the one number now; this fails if a new
    asset lands at a different one, which is the only way a directory of images stays
    uniform.
    """
    expected = 200
    images = [
        image for image in sorted(ROOT.glob("**/assets/**/*.png")) if ".venv" not in image.parts
    ]
    # Only committed figures. `make graph` and `make onnx` write into gitignored
    # directories, and a locally generated file is not something a reader ever sees --
    # failing on it would make the suite depend on which targets someone last ran.
    ignored = subprocess.run(
        ["git", "check-ignore", "--stdin"],
        input="\n".join(str(image) for image in images),
        capture_output=True,
        text=True,
        cwd=ROOT,
    ).stdout.split()
    wrong = []
    for image in images:
        if str(image) in ignored:
            continue
        actual = png_dpi(image)
        if actual != expected:
            wrong.append(f"{image.relative_to(ROOT)} is {actual or 'unset'}")
    assert not wrong, f"figures not at {expected} DPI: " + ", ".join(wrong)
