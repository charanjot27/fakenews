"""Entry point: `python -m fakenews app` (or `streamlit run app/Home.py`)."""
from pathlib import Path

import streamlit as st

st.set_page_config(page_title="SatyaCheck-Text", page_icon="🔎", layout="wide")

import common  # noqa: E402

V = Path(__file__).parent / "views"
pages = {
    "Overview": [st.Page(str(V / "home.py"), title="Home", icon="🏠", default=True),
                 st.Page(str(V / "data_explorer.py"), title="Data explorer", icon="🗂️")],
    "Results": [st.Page(str(V / "comparison.py"), title="Model comparison", icon="📊"),
                st.Page(str(V / "internals.py"), title="Model internals (transparency)", icon="🔬"),
                st.Page(str(V / "hypotheses.py"), title="Hypotheses", icon="🧪"),
                st.Page(str(V / "audit.py"), title="Shortcut audit", icon="🕵️")],
    "Try it": [st.Page(str(V / "live.py"), title="Live checker", icon="✍️"),
               st.Page(str(V / "batch.py"), title="Batch checker", icon="📄"),
               st.Page(str(V / "upload.py"), title="Use your own dataset", icon="⬆️")],
    "Learn": [st.Page(str(V / "concepts.py"), title="Concepts lab", icon="💡"),
              st.Page(str(V / "about.py"), title="About & limitations", icon="ℹ️")],
}
nav = st.navigation(pages)
common.sidebar()
nav.run()
