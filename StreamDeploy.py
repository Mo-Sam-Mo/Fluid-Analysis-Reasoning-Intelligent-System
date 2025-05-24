import os
import time
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from Main import FARIS
from plotly.subplots import make_subplots

faris = FARIS()

if "play_audio" not in st.session_state:
    st.session_state.play_audio = False
if "audio_path" not in st.session_state:
    st.session_state.audio_path = None


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

def play_audio_placeholder(audio_path):
    with st.spinner('🔊 FARIS is analyzing your sample...'):
        time.sleep(2)

    st.success("Analysis complete! Playing audio now...")
    
    # Play audio
    audio_file = open(audio_path, 'rb')
    audio_bytes = audio_file.read()
    st.audio(audio_bytes, format='audio/mp3')

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
        if st.button("🛠️ Sample Diagnostics", use_container_width=True):
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
                value=(min(df['particle_size']), max(df['particle_size']))
            )
        
        # Add some visual flair
        st.markdown("---")
        st.markdown("🔮 *Adjust filters to see real-time updates*")
    
    # Apply filters
    df = df[df["label"].isin(selected_labels)]
    df = df[(df['delta_visc_40'] >= delta_visc_40_range[0]) & (df['delta_visc_40'] <= delta_visc_40_range[1])]
    
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
    st.title("🛠️ FARIS Sample Diagnostics")
    
    # Load data for reference values
    df = load_data()
    
    # Input options
    input_method = st.radio(
        "How would you like to provide the sample?",
        ["Enter values manually", "Upload lab report image"],
        horizontal=True
    )
    
    if input_method == "Enter values manually":
        with st.form("sample_form"):
            st.subheader("Sample Parameters")
            
            # Define feature names
            FEATURE_NAMES = ['Cu', 'Fe', 'Cr', 'Al', 'Si', 'Pb', 'Sn', 'Ni', 'Na', 'B', 'P', 'Zn',
                           'Mo', 'Ca', 'Mg', 'TBN', 'V100', 'V40', 'OXI', 'TAN', 'water_flag', 'antifreeze_flag']
            
            # Organize into two columns for better layout
            col1, col2 = st.columns(2)
            
            with col1:
                # Metal content (ppm)
                st.markdown("### Metal Content (ppm)")
                metal_features = ['Cu', 'Fe', 'Cr', 'Al', 'Si', 'Pb', 'Sn', 'Ni', 'Na']
                metal_values = {}
                for feature in metal_features:
                    metal_values[feature] = st.number_input(
                        f"{feature} (ppm)",
                        min_value=0.0,
                        value=0.0,
                        step=0.1,
                        key=f"metal_{feature}"
                    )
                
                # Viscosity and other physical properties
                st.markdown("### Physical Properties")
                V100 = st.number_input("V100 (cSt)", min_value=0.0, value=10.5)
                V40 = st.number_input("V40 (cSt)", min_value=0.0, value=45.2)
            
            with col2:
                # Additives and other elements
                st.markdown("### Additives and Elements")
                additive_features = ['B', 'P', 'Zn', 'Mo', 'Ca', 'Mg']
                additive_values = {}
                for feature in additive_features:
                    additive_values[feature] = st.number_input(
                        f"{feature} (ppm)",
                        min_value=0.0,
                        value=0.0,
                        step=0.1,
                        key=f"additive_{feature}"
                    )
                
                # Chemical properties
                st.markdown("### Chemical Properties")
                TBN = st.number_input("TBN (mg KOH/g)", min_value=0.0, value=6.5)
                OXI = st.number_input("OXI (Abs/cm)", min_value=0.0, value=0.25)
                TAN = st.number_input("TAN (mg KOH/g)", min_value=0.0, value=1.2)
                
                # Flags
                st.markdown("### Contamination Flags")
                water_flag = st.checkbox("Water Contamination")
                antifreeze_flag = st.checkbox("Antifreeze Contamination")
            
            submitted = st.form_submit_button("Analyze Sample")
            
        if submitted:
            # Prepare the sample data
            sample_data = list(metal_values.values()) + list(additive_values.values()) + [V100, V40, TBN, OXI, TAN, int(water_flag), int(antifreeze_flag)]                
            
            with st.spinner("🔬 Analyzing sample..."):
                time.sleep(2)
                cls, reasoning, audio = faris.predict(sample_data)

                st.success("Analysis Complete!")
                st.markdown(f"### Diagnosis: **{cls}**")
                st.markdown(f"#### Reason: {reasoning}")

                


            # OUTSIDE the form, still inside the "Enter values manually" branch
            if st.session_state.play_audio and st.session_state.audio_path:
                if st.button("🗣️ Ask FARIS for More Insights", help="Get detailed audio explanation", use_container_width=True):
                    play_audio_placeholder('audio/' + audio)

    else:  # Upload lab report image
        uploaded_file = st.file_uploader("Upload Lab Report Image", type=["jpg", "png", "jpeg"])
        
        if uploaded_file is not None:
            # Display the uploaded image
            st.image(uploaded_file, caption="Uploaded Lab Report", width=300)
            
            if st.button("Analyze Report", use_container_width=True):
                # Simulate image analysis
                with st.spinner("👁️ Reading lab report..."):
                    time.sleep(3)
                    
                    # Mock results from image analysis
                    diagnosis = "Normal Wear with Slight Contamination"
                    reason = """The report indicates normal wear metals but shows elevated silicon levels,
                    suggesting possible dirt contamination. The viscosity parameters are within normal range."""
                    
                    st.success("Report Analysis Complete!")
                    st.markdown(f"### Diagnosis: **{diagnosis}**")
                    st.markdown(f"#### Reason: {reason}")
                    
                    # Ask FARIS button
                    st.markdown("---")
                    if st.button("🗣️ Ask FARIS for More Insights", 
                                help="Get detailed audio explanation",
                                use_container_width=True):
                        play_audio_placeholder()
    
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