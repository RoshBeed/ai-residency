"""ML systems portfolio -- one Streamlit frontend over every backend's HTTP API.

This file is the entry point and nothing else: page config, theme, and the navigation
that maps a URL to one of the modules in `views/`. Each of those is a thin client
around one project's already-verified endpoint contract -- no inference logic lives
here, only request/response wiring and display.

Every page renders a real trace of what the backend actually did to produce its answer
(which words were recognised, per-token probabilities, a live fine-tuning loss curve,
the PPO policy's sampling candidates at each generation step) rather than only the
final result. That data comes straight from each project's own API response; nothing
here recomputes or simulates it.

Routing is `st.navigation`, not a hand-rolled state machine. What it replaced was 892
lines of `if _PAGE == ...` over the nine services plus a session-state and query-param
dance to keep the URL and the sidebar agreeing -- all of it at module scope, which is
why importing anything from this file used to start a script run and made the page
flow untestable. Streamlit owns the URL, the browser history and the sidebar now, and
each page is an ordinary function that can be imported and called.
"""

from collections.abc import Callable
from importlib import import_module
from typing import Literal

import streamlit as st

from projects import PAGES, PROJECTS
from theme import LOGO_URI, PALETTE_CSS
from views import home

st.set_page_config(page_title="ML Systems Portfolio", page_icon="🧩", layout="wide")
st.logo(LOGO_URI)
st.markdown(PALETTE_CSS, unsafe_allow_html=True)


def _renderer(service: str) -> Callable[[], None]:
    """The `render()` of the view module for one service.

    A callable, not the `st.Page("views/x.py")` file form. Given a path Streamlit
    executes the file as a script, which would put every page back to running its
    flow at import -- the thing this layout exists to stop. A function is imported,
    called by the router, and can equally be called by a test.
    """
    module = import_module(f"views.{service.replace('-', '_')}")
    render: Callable[[], None] = module.render
    return render


# Derived from PAGES rather than listed, so a service appears here the moment it is
# configured. `url_path` is the service name, so the links this app has always
# produced keep working and a page stays shareable by URL.
service_pages = {
    service: st.Page(
        _renderer(service),
        title=label,
        icon=PROJECTS[service]["icon"],
        url_path=service,
    )
    for service, label in PAGES.items()
}

# One configured page is the landing page: there is no gallery worth showing, and no
# navigation to offer. `AI_RESIDENCY_PAGES` is what makes the one hub image serve a
# single system without a second UI existing anywhere.
if len(service_pages) == 1:
    pages = list(service_pages.values())
    position: Literal["sidebar", "hidden", "top"] = "hidden"
else:

    def show_home() -> None:
        """Adapter: `st.Page` wants a no-argument callable, home needs the targets."""
        home.render(service_pages)

    pages = [
        st.Page(show_home, title="Home", icon="🏠", url_path="home", default=True),
        *service_pages.values(),
    ]
    position = "sidebar"

with st.sidebar:
    st.caption("[GitHub →](https://github.com/RoshBeed/ai-residency/tree/main/services/frontend)")
    st.caption("[LinkedIn →](https://linkedin.com/in/RoshBeed)")

st.navigation(pages, position=position).run()
