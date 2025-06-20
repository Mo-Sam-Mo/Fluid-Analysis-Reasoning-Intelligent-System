import streamlit as st
from PIL import Image
from StreamlitDeploy import dashboard_page, analysis_page, diagnostics_page

# =========================
# Main App Layout
# =========================
st.set_page_config(
    page_title="🛢️ FARIS Fluid Analysis",
    layout="wide",
    initial_sidebar_state="expanded",
    )

# Sidebar Branding
st.sidebar.image("Logo.jpg", use_column_width="auto")
st.sidebar.markdown("""
## 🛢️ FARIS Dashboard
*Fluid Analysis & Reasoning Intelligence System*

---
""")

# ⚡️ Dark/Light Mode Toggle
dark_mode = st.sidebar.toggle("🌗 Dark Mode", value=False)
st.session_state["dark_mode"] = dark_mode

# =========================
# Apply Styles Based on Mode
# =========================
if dark_mode:
    st.markdown("""
    <style>
        /* Main app and sidebar background */
        body, .stApp {
            background-color: #1e1e1e !important;
            color: #eaeaea !important;
        }
        .stSidebar, [data-testid="stSidebar"] {
            background-color: #2c2c2c !important;
        }

        /* Text elements */
        .stMarkdown, .stTable, .stDataFrame, .stText, .stHeader, .stSubheader, .stTitle, .stCaption {
            color: #eaeaea !important;
        }

        /* Widget Labels and Buttons */
        label, .stButton button, .stSelectbox, .stNumberInput {
            color: #eaeaea !important;
        }

        /* Links */
        a {
            color: #1abc9c !important;
        }

        /* Keep plotly chart text readable (Plotly uses its own theme) */
        .js-plotly-plot .main-svg .infolayer text {
            fill: #eaeaea !important;
        }
    </style>
    """, unsafe_allow_html=True)

else:
    st.markdown("""
    <style>
        body, .stApp {
            background-color: #ffffff !important;
            color: #000000 !important;
        }
    </style>
    """, unsafe_allow_html=True)

# =========================
# Page Navigation
# =========================
st.sidebar.markdown("---\n**Choose a page:**")
page = st.sidebar.radio(
    "Navigation",
    ["🏠 Dashboard", "🔍 Analysis", "🛠️ Diagnostics"]
)

# =========================
# Route to Selected Page
# =========================
if page == "🏠 Dashboard":
    dashboard_page()
elif page == "🔍 Analysis":
    analysis_page()
elif page == "🛠️ Diagnostics":
    diagnostics_page()

# =========================
# Footer
# =========================
st.markdown("""
---
<div style='text-align: center; font-size: 14px; color: #888;'>
    Developed by the FARIS Team | © 2025
</div>
""", unsafe_allow_html=True)
