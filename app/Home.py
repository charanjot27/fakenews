"""Entry point: `python -m fakenews app` (or `streamlit run app/Home.py`)."""
from pathlib import Path

import streamlit as st

st.set_page_config(page_title="SatyaCheck-Text", page_icon=":material/fact_check:", layout="wide")
st.markdown("""<style>
#MainMenu, footer, [data-testid="stToolbar"], [data-testid="stDecoration"] {display:none;}
h1, h2, h3 {font-weight:600; letter-spacing:-0.02em;}
h1 {font-size:1.9rem;}
[data-testid="stAlert"], [data-testid="stExpander"], [data-testid="stVerticalBlockBorderWrapper"] {border-radius:4px;}
[data-testid="stImage"] img {border-radius:4px; border:1px solid #222;}
.stButton button, .stDownloadButton button {border-radius:4px;}
</style>""", unsafe_allow_html=True)

import common  # noqa: E402

V = Path(__file__).parent / "views"
pages = {
    "Overview": [st.Page(str(V / "home.py"), title="Home", icon=":material/home:", default=True),
                 st.Page(str(V / "data_explorer.py"), title="Data explorer", icon=":material/table_view:")],
    "Results": [st.Page(str(V / "comparison.py"), title="Model comparison", icon=":material/leaderboard:"),
                st.Page(str(V / "internals.py"), title="Model internals (transparency)", icon=":material/memory:"),
                st.Page(str(V / "hypotheses.py"), title="Hypotheses", icon=":material/science:"),
                st.Page(str(V / "audit.py"), title="Shortcut audit", icon=":material/policy:")],
    "Try it": [st.Page(str(V / "live.py"), title="Live checker", icon=":material/edit_note:"),
               st.Page(str(V / "batch.py"), title="Batch checker", icon=":material/description:"),
               st.Page(str(V / "upload.py"), title="Use your own dataset", icon=":material/upload:")],
    "Learn": [st.Page(str(V / "concepts.py"), title="Concepts lab", icon=":material/school:"),
              st.Page(str(V / "about.py"), title="About & limitations", icon=":material/info:")],
}
nav = st.navigation(pages)
common.sidebar()
nav.run()
