import streamlit as st
import json
import tempfile
import pandas as pd
import matplotlib.pyplot as plt
import os
from audio_processor import extract_fingerprint

st.set_page_config(page_title="Zapptain America", layout="wide")

# --- Load Database ---
@st.cache_data
def load_database():
    try:
        with open('database.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

db = load_database()

# --- Matching Logic ---
def identify_song(query_hashes, database):
    best_match = "Unknown Song"
    highest_score = 0
    best_histogram = []

    for song_name, db_hashes in database.items():
        # Convert db hashes into a dictionary for faster lookup
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
            # Find the most common offset (the spike in the histogram)
            counts = pd.Series(offsets).value_counts()
            max_count = counts.iloc[0] if not counts.empty else 0
            
            if max_count > highest_score:
                highest_score = max_count
                best_match = song_name
                best_histogram = offsets

    return best_match, best_histogram

# --- UI Setup ---
st.title("🎵 Zapptain America: Audio Identifier")

if not db:
    st.error("database.json not found! Please ensure it is uploaded to the repository.")
    st.stop()

tab1, tab2 = st.tabs(["Single-Clip Mode", "Batch Mode"])

# =========================================
# TAB 1: SINGLE CLIP MODE
# =========================================
with tab1:
    st.header("Identify a Single Clip")
    uploaded_file = st.file_uploader("Upload an audio snippet", type=['mp3', 'wav'], key="single")

    if uploaded_file is not None:
        st.audio(uploaded_file)
        
        if st.button("Identify Song", type="primary"):
            with st.spinner("Analyzing spectral signatures..."):
                # Save uploaded file temporarily to pass to librosa
                with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_path = tmp_file.name

                # Process
                query_hashes, Sxx_db, t, f, p_time, p_freq = extract_fingerprint(tmp_path)
                os.remove(tmp_path) # Cleanup

                if not query_hashes:
                    st.error("Could not extract audio features.")
                else:
                    match_name, histogram = identify_song(query_hashes, db)
                    
                    st.success(f"### 🎶 Matched Song: **{match_name}**")
                    
                    st.divider()
                    st.subheader("Diagnostic Visuals")
                    
                    col1, col2 = st.columns(2)
                    
                    # Visual 1: Spectrogram & Constellation
                    with col1:
                        fig1, ax1 = plt.subplots(figsize=(8, 5))
                        ax1.pcolormesh(t, f, Sxx_db, shading='gouraud', cmap='magma', alpha=0.8)
                        ax1.scatter(t[p_time], f[p_freq], c='cyan', s=5, marker='o')
                        ax1.set_title("Spectrogram & Peak Constellation")
                        ax1.set_ylabel("Frequency Bin")
                        ax1.set_xlabel("Time Bin")
                        st.pyplot(fig1)
                        
                    # Visual 2: Offset Histogram
                    with col2:
                        fig2, ax2 = plt.subplots(figsize=(8, 5))
                        if histogram:
                            ax2.hist(histogram, bins=100, color='green')
                        ax2.set_title(f"Time Offset Histogram (Match: {match_name})")
                        ax2.set_xlabel("Time Offset")
                        ax2.set_ylabel("Matching Hashes")
                        st.pyplot(fig2)

# =========================================
# TAB 2: BATCH MODE
# =========================================
with tab2:
    st.header("Batch Process Queries")
    batch_files = st.file_uploader("Upload multiple query files", type=['mp3', 'wav'], accept_multiple_files=True, key="batch")

    if batch_files and st.button("Process Batch"):
        with st.spinner("Processing multiple files..."):
            results = []
            
            # Progress bar
            progress_bar = st.progress(0)
            for idx, b_file in enumerate(batch_files):
                with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as tmp_file:
                    tmp_file.write(b_file.getvalue())
                    tmp_path = tmp_file.name
                    
                q_hashes, _, _, _, _, _ = extract_fingerprint(tmp_path)
                os.remove(tmp_path)
                
                if q_hashes:
                    pred_name, _ = identify_song(q_hashes, db)
                else:
                    pred_name = "Error"
                    
                results.append({
                    "filename": b_file.name,
                    "prediction": pred_name
                })
                progress_bar.progress((idx + 1) / len(batch_files))
                
            # Create DataFrame exactly as required by rubric
            df = pd.DataFrame(results)
            csv_data = df.to_csv(index=False).encode('utf-8')
            
            st.success("Batch processing complete!")
            st.dataframe(df)
            
            st.download_button(
                label="📥 Download results.csv",
                data=csv_data,
                file_name="results.csv",
                mime="text/csv"
            )