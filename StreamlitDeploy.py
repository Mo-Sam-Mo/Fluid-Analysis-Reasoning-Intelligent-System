import os
import time
import base64
import pandas as pd
from PIL import Image
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from Reasoning import Reasoning_Model
from plotly.subplots import make_subplots
import pickle
import gspread
import requests
from oauth2client.service_account import ServiceAccountCredentials
from Reasoning import Reasoning_Model
from Main import FARIS


faris = FARIS()


# ----- Functions -----
@st.cache_data
def load_data():
    df = pd.read_csv("cleaned_synthetic_oil_data.csv")
    return df

def plot_feature_distribution(df, feature_col, target_col):
    fig = px.histogram(
        df,
        x=feature_col,
        color=target_col,
        marginal="violin",  # Adds violin plot on top
        hover_data=df.columns,
        title=f'Distribution of {feature_col} by {target_col}',
        template="plotly_dark",
        nbins=50,
        opacity=0.7
    )
    
    # Update layout for better visualization
    fig.update_layout(
        xaxis_title=feature_col,
        yaxis_title="Density",
        legend_title=target_col,
        hovermode="x unified",
        barmode="overlay"
    )
    
    # Add KDE curves for each class
    for cls in df[target_col].unique():
        fig.add_trace(
            go.Violin(
                x=df[df[target_col] == cls][feature_col],
                name=str(cls),
                box_visible=True,
                meanline_visible=True,
                fillcolor='rgba(0,0,0,0)',
                line_color=px.colors.qualitative.Plotly[len(fig.data) % len(px.colors.qualitative.Plotly)],
                opacity=0.6
            )
        )
    
    # Customize hover information
    fig.update_traces(
        hovertemplate=f"<b>{feature_col}</b>: %{{x}}<br><b>Count</b>: %{{y}}"
    )
    
    return fig

def create_interactive_kpi(value, label, delta=None):
    kpi_html = f"""
    <div style="
        padding: 20px;
        border-radius: 10px;
        background: linear-gradient(145deg, #2c3e50, #34495e);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        transition: all 0.3s ease;
        margin: 10px;
        text-align: center;
        color: white;
    "
    onmouseover="this.style.transform='scale(1.05)'; this.style.boxShadow='0 8px 16px rgba(0,0,0,0.3)';"
    onmouseout="this.style.transform='scale(1)'; this.style.boxShadow='0 4px 8px rgba(0,0,0,0.2)';">
        <h2 style="margin: 0; font-size: 2.5rem; color: #1abc9c">{value}</h2>
        <p style="margin: 5px 0 0; font-size: 1rem">{label}</p>
        {f'<p style="margin: 5px 0 0; font-size: 0.8rem; color: {"#2ecc71" if delta >= 0 else "#e74c3c"}">{delta:+}%</p>' if delta is not None else ''}
    </div>
    """
    return kpi_html

def apply_plot_effects(fig):
    fig.update_layout(
        hovermode="x unified",
        hoverlabel=dict(
            bgcolor="rgba(0,0,0,0.7)",
            font_size=14,
            font_color="white"
        ),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=False),
        transition={'duration': 500},
    )
    return fig

def play_audio_placeholder():
    with st.spinner('🔊 FARIS is analyzing your sample...'):
        time.sleep(2)
        st.success("Analysis complete! (Audio placeholder - would play insights.mp3 in production)")

# ----- Page Config -----
st.set_page_config(
    page_title="🛢️ FARIS Fluid Analysis",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----- Page 1: Dashboard -----
def dashboard_page():
    st.title("🛢️ FARIS Dashboard")
    
    # Logo and header
    col1, col2 = st.columns([1, 4])
    with col1:
        try:
            logo = Image.open("FARIS.jpg")
            st.image(logo, width=150)
        except:
            st.image(Image.new('RGB', (150, 150), color='#2c3e50'))
    
    with col2:
        st.markdown("""
        <style>
            .title-text {
                font-size: 2.5rem !important;
                font-weight: bold;
                color: #1abc9c;
                margin-bottom: 0;
            }
            .subtitle-text {
                font-size: 1.2rem !important;
                color: #7f8c8d;
                margin-top: 0;
            }
        </style>
        <p class="title-text">Fluid Analysis & Reasoning Intelligence System</p>
        <p class="subtitle-text">Real-time oil analysis diagnostics</p>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Interactive KPIs
    df = load_data()
    st.markdown("""
    <style>
        .kpi-container {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 15px;
            margin-bottom: 30px;
        }
        @media (max-width: 1200px) {
            .kpi-container {
                grid-template-columns: repeat(2, 1fr);
            }
        }
    </style>
    <div class="kpi-container">
    """, unsafe_allow_html=True)
    
    # Create 4 columns for KPIs
    cols = st.columns(4)
    
    with cols[0]:
        st.markdown(create_interactive_kpi(
            len(df), 
            "Total Samples", 
            delta=2.5
        ), unsafe_allow_html=True)
    
    with cols[1]:
        st.markdown(create_interactive_kpi(
            df["label"].nunique(), 
            "Unique Diagnoses",
            delta=1.8
        ), unsafe_allow_html=True)
    
    with cols[2]:
        st.markdown(create_interactive_kpi(
            df.shape[1] - 1, 
            "Features Analyzed",
            delta=3.2
        ), unsafe_allow_html=True)
    
    with cols[3]:
        st.markdown(create_interactive_kpi(
            f"{df['delta_visc_40'].mean():.1f}", 
            "Avg delta_visc_40 (cSt)",
            delta=-0.5
        ), unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Mini visualizations
    st.subheader("📈 Quick Insights")
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.pie(df, names='label', title='Diagnosis Distribution')
        fig = apply_plot_effects(fig)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.histogram(df, x='delta_visc_40', nbins=20, title='delta_visc_40 Distribution')
        fig = apply_plot_effects(fig)
        st.plotly_chart(fig, use_container_width=True)
    
    # Navigation buttons
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔍 Explore Full Analysis", use_container_width=True):
            st.session_state.page = "analysis"
    with col2:
        if st.button("🛠️ Equipment Diagnostics", use_container_width=True):
            st.session_state.page = "diagnostics"

# ----- Page 2: Analysis -----
def analysis_page():
    st.title("🔍 FARIS Advanced Analysis")
    
    # Enhanced sidebar with creative filter
    with st.sidebar:
        st.header("🎛️ Smart Filters")
        
        # Dynamic filter
        df = load_data()
        labels = df["label"].unique()
        selected_labels = st.multiselect(
            "Select Diagnoses:",
            labels,
            default=labels,
            help="Filter by diagnosis type"
        )
        
        # New creative filter - delta_visc_40 range with animation
        delta_visc_40_range = st.slider(
            "delta_visc_40 Range (cSt):",
            min_value=float(df['delta_visc_40'].min()),
            max_value=float(df['delta_visc_40'].max()),
            value=(float(df['delta_visc_40'].quantile(0.25)), 
            float(df['delta_visc_40'].quantile(0.75))),
            step=0.1,
            key="delta_visc_40"
        )
        
        # Particle size distribution filter
        if 'particle_size' in df.columns:
            particle_size = st.select_slider(
                "Particle Size Range (µm):",
                options=sorted(df['particle_size'].unique()),
                value=(min(df['particle_size']), max(df['particle_size'])))
        
        # Add some visual flair
        st.markdown("---")
        st.markdown("🔮 *Adjust filters to see real-time updates*")
    
    # Apply filters
    df = df[df["label"].isin(selected_labels)]
    df = df[(df['delta_visc_40'] >= delta_visc_40_range[0]) & 
            (df['delta_visc_40'] <= delta_visc_40_range[1])]
    
    # Interactive visualizations
    st.subheader("📊 Interactive Data Explorer")
    
    # Feature selector with animation
    numerical_cols = [col for col in df.select_dtypes(include='number').columns if col != 'label']
    selected_features = st.multiselect(
        "Select features to visualize:",
        numerical_cols,
        default=numerical_cols[:3]
    )
    
    if selected_features:
        fig = px.scatter_matrix(
            df,
            dimensions=selected_features,
            color="label",
            title="Feature Relationships",
            hover_name="label",
            hover_data=df.columns
        )
        fig = apply_plot_effects(fig)
        st.plotly_chart(fig, use_container_width=True)
    
    # Enhanced feature distribution view
    st.subheader("📈 Feature Distributions")
    selected_feature = st.selectbox(
        "Select a feature to analyze:",
        numerical_cols,
        index=0
    )
    
    col1, col2 = st.columns(2)
    with col1:
        fig = px.box(df, x="label", y=selected_feature, color="label")
        fig = apply_plot_effects(fig)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.violin(df, x="label", y=selected_feature, box=True, points="all", color="label")
        fig = apply_plot_effects(fig)
        st.plotly_chart(fig, use_container_width=True)
    
    # Back button
    if st.button("← Back to Dashboard", type="secondary"):
        st.session_state.page = "dashboard"

# ----- Page 3: Diagnostics -----
def diagnostics_page():
    st.title("🔍 Equipment Failure Prediction Dashboard")
    st.markdown("This app predicts equipment failure class using either manual inputs or image-based oil analysis.")
    
    # --- Reset button ---
    if st.button("🔁 Reset Inputs"):
        for key in list(st.session_state.keys()):
            if key not in ['page', 'input_mode']:
                del st.session_state[key]
        st.experimental_rerun()
    
    
    # --- Input Columns ---
    input_columns = [
        'Cu', 'Fe', 'Cr', 'Al', 'Si', 'Pb', 'Sn', 'Ni', 'Na', 'B',
        'P', 'Zn', 'Mo', 'Ca', 'Mg', 'TBN', 'V100', 'V40',
        'OXI', 'TAN', 'water_flag', 'antifreeze_flag'
    ]
    
    # --- Input Mode Selection ---
    if "input_mode" not in st.session_state:
        st.session_state.input_mode = "📝 Manual Entry"
    input_mode = st.radio("Choose Input Mode", ["📝 Manual Entry", "📷 Upload Image"],
                          index=["📝 Manual Entry", "📷 Upload Image"].index(st.session_state.input_mode),
                          horizontal=True)
    st.session_state.input_mode = input_mode
    
    # --- Input Data Placeholder ---
    if "input_data" not in st.session_state:
        st.session_state.input_data = None
    
# --- Manual Entry ---
    if input_mode == "📝 Manual Entry":
        with st.form("manual_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 🧪 Metal Content (ppm)")
                Cu = st.number_input("Copper (Cu)", 0.0, 100.0, 5.0)
                Fe = st.number_input("Iron (Fe)", 0.0, 100.0, 6.0)
                Cr = st.number_input("Chromium (Cr)", 0.0, 100.0, 0.0)
                Al = st.number_input("Aluminum (Al)", 0.0, 100.0, 1.0)
                Si = st.number_input("Silicon (Si)", 0.0, 100.0, 3.0)
                Pb = st.number_input("Lead (Pb)", 0.0, 100.0, 18.0)
                Sn = st.number_input("Tin (Sn)", 0.0, 100.0, 1.0)
                Ni = st.number_input("Nickel (Ni)", 0.0, 100.0, 0.0)
                Na = st.number_input("Sodium (Na)", 0.0, 100.0, 8.0)

                st.markdown("### 🧫 Physical Properties")
                V100 = st.number_input("Viscosity at 100°C (V100, cSt)", 0.0, 100.0, 10.0)
                V40 = st.number_input("Viscosity at 40°C (V40, cSt)", 0.0, 200.0, 150.0)

            with col2:
                st.markdown("### 🧪 Additives and Elements (ppm)")
                B = st.number_input("Boron (B)", 0.0, 100.0, 1.0)
                P = st.number_input("Phosphorus (P)", 0.0, 100.0, 15.0)
                Zn = st.number_input("Zinc (Zn)", 0.0, 100.0, 15.0)
                Mo = st.number_input("Molybdenum (Mo)", 0.0, 100.0, 0.0)
                Ca = st.number_input("Calcium (Ca)", 0.0, 500.0, 256.0)
                Mg = st.number_input("Magnesium (Mg)", 0.0, 100.0, 0.0)

                st.markdown("### 🔬 Chemical Properties")
                TBN = st.number_input("Total Base Number (TBN, mg KOH/g)", 0.0, 10.0, 8.5)
                OXI = st.number_input("Oxidation (OXI, Abs/cm)", 0.0, 100.0, 0.0)
                TAN = st.number_input("Total Acid Number (TAN, mg KOH/g)", 0.0, 2.0, 0.04)

                st.markdown("### ⚠️ Contamination Flags")
                water_flag = st.checkbox("Water Present?", value=True)
                antifreeze_flag = st.checkbox("Antifreeze Present?", value=False)

            submit_manual = st.form_submit_button("🔍 Predict")

            if submit_manual:
                st.session_state.input_data = [
                    Cu, Fe, Cr, Al, Si, Pb, Sn, Ni, Na, B, P, Zn, Mo, Ca, Mg,
                    TBN, V100, V40, OXI, TAN, 
                    int(water_flag), int(antifreeze_flag)
                ]

    
    # --- Upload Image Mode (To be added later) ---
    elif input_mode == "📷 Upload Image":
        st.info("📷 Image mode will be supported soon. Stay tuned!")
    
    # --- Prediction Section ---
    if st.session_state.input_data is not None:
        input_data = st.session_state.input_data

        prediction, reasoning_text, audio_file_path = faris.predict(input_data)

        
        st.success(f"🔧 Predicted Failure Class: *{prediction}*")
    
        # Save to Google Sheets
        try:
            scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
            creds_dict = st.secrets["gcp_service_account"]
            creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
            sheet = gspread.authorize(creds).open("Equipment_Predictions").sheet1
            row = input_data + [prediction]
            sheet.append_row(row)
            st.info("✅ Saved to Google Sheet.")
        except Exception as e:
            st.warning(f"⚠ Could not save to Google Sheet: {e}")
    
        # 🧠 LLM Reasoning
        try:
            st.markdown("### 💬 LLM Reasoning")
            st.info(reasoning_text)
        except Exception as e:
            st.warning(f"⚠ Could not generate reasoning: {e}")
    
        # 🔊 Audio Explanation
        if st.button("🎧 Listen to Reasoning"):
            try:
                with open('audio/' + audio_file_path, "rb") as audio_file:
                    st.audio(audio_file.read(), format="audio/mp3")
            except Exception as e:
                st.error(f"❌ Audio Error: {e}")
            
            # Back button
            if st.button("← Back to Dashboard", type="secondary"):
                st.session_state.page = "dashboard"

# ----- Main App -----
def main():
    # Initialize session state
    if "page" not in st.session_state:
        st.session_state.page = "dashboard"
    # Page navigation
    if st.session_state.page == "dashboard":
        dashboard_page()
    elif st.session_state.page == "analysis":
        analysis_page()
    elif st.session_state.page == "diagnostics":
        diagnostics_page()

if __name__ == "__main__":
    main()