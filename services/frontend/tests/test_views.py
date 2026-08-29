"""Every page, actually rendered.

The nine service pages used to be one 892-line `if/elif` chain at module scope in
`app.py`, which meant the only way to execute a page was to start Streamlit and click
to it. Nothing here was covered: a typo in a page's response handling reached the
browser before it reached a test.

Each page is a `render()` in its own module now, so Streamlit's own `AppTest` can run
it headlessly. These are smoke tests rather than assertions about layout -- what they
catch is a page that raises: a bad import, a renamed helper, a component called with
the wrong shape.

Readiness is patched rather than left to the environment. Both paths matter and they
run different code: an unready backend renders one message and stops, a ready one
renders the whole demo. Left unpatched these tests would quietly test whichever
backends happened to be running on the developer's machine.
"""

from collections.abc import Iterator
from contextlib import contextmanager
from importlib import import_module
from unittest.mock import patch

import pytest
from streamlit.testing.v1 import AppTest

import client
import shared
from projects import ALL_PAGES

VIEW_MODULES = sorted(service.replace("-", "_") for service in ALL_PAGES)


@contextmanager
def backend(*, ready: bool) -> Iterator[None]:
    """Pretend every backend is up, or down, without one being either.

    `shared.is_ready` rather than `client.is_ready`: `shared` imported the name, so
    that binding is the one the page framing actually calls. Patching the definition
    in `client` would leave the framing pointing at the original.
    """
    unreachable = client.Result(status="failed", message="stubbed")
    with (
        patch.object(shared, "is_ready", return_value=ready),
        patch.object(client, "call", return_value=unreachable),
    ):
        yield


def run_render(module_name: str) -> AppTest:
    """Run one view's `render()` as a Streamlit script and return the finished run.

    `from_string`, not `from_function`: the latter writes only the function's own
    source into a temporary script, so the module's imports are gone and every page
    fails on `service_page` being undefined -- a failure about the harness, not the
    page.
    """
    app = AppTest.from_string(
        f"from views.{module_name} import render\nrender()", default_timeout=60
    )
    app.run()
    return app


@pytest.mark.parametrize("module_name", VIEW_MODULES)
def test_a_page_whose_backend_is_down_explains_itself(module_name: str) -> None:
    """One message and a stop, rather than an empty demo that fails on first use."""
    with backend(ready=False):
        app = run_render(module_name)
    assert not app.exception, f"views/{module_name}.py raised: {[e.value for e in app.exception]}"
    assert app.error, f"views/{module_name}.py rendered no not-ready message"
    assert "isn't ready" in app.error[0].value


@pytest.mark.parametrize("module_name", VIEW_MODULES)
def test_a_page_whose_backend_is_up_renders_its_demo(module_name: str) -> None:
    """The path with the page's own code in it: inputs, controls and write-up.

    A page that raises here is one that reached the browser broken -- which is how
    every frontend bug in this repo was previously found.
    """
    with backend(ready=True):
        app = run_render(module_name)
    assert not app.exception, f"views/{module_name}.py raised: {[e.value for e in app.exception]}"
    assert app.subheader, f"views/{module_name}.py rendered no demo section"
    assert any("Try it out" in block.value for block in app.subheader)


def test_the_home_page_renders_a_card_for_every_service() -> None:
    """The gallery is the only page with no backend of its own, so it always renders."""
    app = AppTest.from_string(
        "import streamlit as st\n"
        "from projects import ALL_PAGES\n"
        "from views import home\n"
        "targets = {s: st.Page(lambda: None, title=s, url_path=s) for s in ALL_PAGES}\n"
        "home.render(targets)\n",
        default_timeout=60,
    )
    app.run()
    assert not app.exception, [e.value for e in app.exception]
    cards = "".join(block.value for block in app.markdown)
    for service, label in ALL_PAGES.items():
        assert label in cards, f"{service} has no card on the home page"


def test_every_page_has_a_view_module() -> None:
    """`app.py` resolves these by name, so a missing module is an ImportError at boot."""
    for module_name in VIEW_MODULES:
        module = import_module(f"views.{module_name}")
        assert callable(module.render), f"views/{module_name}.py has no render()"
