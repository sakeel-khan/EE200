import streamlit as st
import json
import tempfile
import pandas as pd
import matplotlib.pyplot as plt
import os
from audio_processor import extract_fingerprint

# ==========================================
# PAGE CONFIGURATION & CUSTOM CSS
# ==========================================
st.set_page_config(page_title="Zapptain America", page_icon="🎧", layout="wide", initial_sidebar_state="expanded")

# Custom CSS for a sleeker look
st.markdown("""
    <style>
    /* Style the main metric box */
    [data-testid="stMetricValue"] {
        font-size: 2.5rem !important;
        color: #1DB954 !important; /* Spotify Green */
    }
    /* Style the upload box */
    .stFileUploader {
        border-radius: 15px;
        padding: 10px;
    }
    /* Style buttons */
    .stButton>button {
        border-radius: 20px;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: scale(1.02);
    }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# DATABASE & LOGIC
# ==========================================
@st.cache_data
def load_database():
    try:
        with open('database.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

db = load_database()

def identify_song(query_hashes, database):
    """Upgraded to return a confidence score (raw hash matches)"""
    best_match = "Unknown Song"
    highest_score = 0
    best_histogram = []

    for song_name, db_hashes in database.items():
        db_dict = {}
        for h_key, t_offset in db_hashes:
            if h_key not in db_dict:
                db_dict[h_key] = []
            db_dict[h_key].append(t_offset)
            
        offsets = []
        for q_key, q_time in query_hashes:
            if q_key in db_dict:
                for db_time in db_dict[q_key]:
                    offsets.append(db_time - q_time)
                    
        if offsets:
            counts = pd.Series(offsets).value_counts()
            max_count = counts.iloc[0] if not counts.empty else 0
            
            if max_count > highest_score:
                highest_score = max_count
                best_match = song_name
                best_histogram = offsets

    return best_match, best_histogram, highest_score

# ==========================================
# SIDEBAR NAVIGATION
# ==========================================
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3659/3659784.png", width=100) # Generic sound wave icon
    st.title("Zapptain America")
    st.caption("v2.0 - Audio Fingerprinting Engine")
    st.divider()
    
    # Use radio buttons for navigation instead of tabs
    app_mode = st.radio("Navigation", ["🔍 Single-Clip Mode", "📂 Batch Processing Mode"])
    
    st.divider()
    st.success(f"📦 Database Loaded: **{len(db)} songs** indexed.")

if not db:
    st.error("⚠️ database.json not found! Please ensure your database is built and located in the main directory.")
    st.stop()

# ==========================================
# MODE 1: SINGLE CLIP IDENTIFIER
# ==========================================
if app_mode == "🔍 Single-Clip Mode":
    st.header("Audio Identification Engine")
    st.write("Upload a noisy or clean audio snippet. The system will extract spectral hashes and find the best match.")
    
    uploaded_file = st.file_uploader("Drop an audio file here (.mp3, .wav)", type=['mp3', 'wav'])

    if uploaded_file is not None:
        # Create two columns for the top section
        top_col1, top_col2 = st.columns([1, 2])
        
        with top_col1:
            st.audio(uploaded_file)
            analyze_button = st.button("🚀 Analyze & Identify", type="primary", use_container_width=True)
            
        if analyze_button:
            with st.spinner("Extracting frequencies and matching hashes..."):
                with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_path = tmp_file.name

                query_hashes, Sxx_db, t, f, p_time, p_freq = extract_fingerprint(tmp_path)
                os.remove(tmp_path) 

                if not query_hashes:
                    st.error("Audio too short or silent to extract features.")
                else:
                    match_name, histogram, score = identify_song(query_hashes, db)
                    
                    # Display results in a highly visible metric card
                    with top_col2:
                        st.metric(label="Identification Result", value=match_name, delta=f"{score} aligned hashes")

                    st.divider()
                    
                    # Group diagnostic visuals inside an expander to keep the UI clean
                    st.subheader("Diagnostic Telemetry")
                    with st.expander("View Spectral Analysis & Match Data", expanded=True):
                        col1, col2 = st.columns(2)
                        
                        # Set matplotlib style to a darker theme to match the app
                        plt.style.use('dark_background')
                        
                        with col1:
                            fig1, ax1 = plt.subplots(figsize=(8, 5))
                            ax1.pcolormesh(t, f, Sxx_db, shading='gouraud', cmap='magma', alpha=0.8)
                            ax1.scatter(t[p_time], f[p_freq], c='#00FF00', s=15, marker='x', linewidths=1)
                            ax1.set_title("Spectrogram & Peak Constellation")
                            ax1.set_ylabel("Frequency (Hz)")
                            ax1.set_xlabel("Time (s)")
                            # Format plot to look modern
                            ax1.grid(False)
                            fig1.patch.set_alpha(0.0)
                            ax1.patch.set_alpha(0.0)
                            st.pyplot(fig1)
                            
                        with col2:
                            fig2, ax2 = plt.subplots(figsize=(8, 5))
                            if histogram:
                                ax2.hist(histogram, bins=100, color='#1DB954', edgecolor='black')
                            ax2.set_title("Time Offset Histogram")
                            ax2.set_xlabel("Time Offset (bins)")
                            ax2.set_ylabel("Matched Hashes")
                            fig2.patch.set_alpha(0.0)
                            ax2.patch.set_alpha(0.0)
                            st.pyplot(fig2)

# ==========================================
# MODE 2: BATCH PROCESSING
# ==========================================
elif app_mode == "📂 Batch Processing Mode":
    st.header("Batch Query Processing")
    st.write("Upload multiple query clips simultaneously to generate your required `results.csv`.")
    
    batch_files = st.file_uploader("Select multiple files", type=['mp3', 'wav'], accept_multiple_files=True)

    if batch_files and st.button("⚙️ Process Batch", type="primary"):
        results = []
        
        # Use Streamlit's built-in progress components
        progress_text = "Processing files. Please wait."
        my_bar = st.progress(0, text=progress_text)
        
        for idx, b_file in enumerate(batch_files):
            # Update progress bar
            my_bar.progress((idx) / len(batch_files), text=f"Processing {b_file.name}...")
            
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as tmp_file:
                tmp_file.write(b_file.getvalue())
                tmp_path = tmp_file.name
                
            q_hashes, _, _, _, _, _ = extract_fingerprint(tmp_path)
            os.remove(tmp_path)
            
            if q_hashes:
                pred_name, _, _ = identify_song(q_hashes, db)
            else:
                pred_name = "Error_No_Features"
                
            results.append({
                "filename": b_file.name,
                "prediction": pred_name
            })
            
        my_bar.progress(1.0, text="Batch processing complete!")
            
        df = pd.DataFrame(results)
        csv_data = df.to_csv(index=False).encode('utf-8')
        
        st.success(f"Successfully processed {len(results)} files.")
        
        # Show data cleanly
        st.dataframe(df, use_container_width=True)
        
        st.download_button(
            label="📥 Download results.csv",
            data=csv_data,
            file_name="results.csv",
            mime="text/csv",
            type="primary"
        )