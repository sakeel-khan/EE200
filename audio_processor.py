import numpy as np
import librosa
from scipy import signal
import scipy.ndimage as ndimage

def extract_fingerprint(file_path):
    """Loads audio, generates spectrogram, finds peaks, and creates hashes."""
    try:
        # sr=None preserves the original sample rate of the mp3/wav
        audio, sr = librosa.load(file_path, sr=None, mono=True)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return [], None, None, None, None, None

    # 1. Spectrogram
    window_size = 1024
    f, t, Sxx = signal.spectrogram(audio, sr, nperseg=window_size)
    Sxx_db = 10 * np.log10(Sxx + 1e-10)

    # 2. Find Peaks (Constellation)
    neighborhood_size = 20
    local_max = ndimage.maximum_filter(Sxx_db, size=neighborhood_size) == Sxx_db
    
    threshold = np.percentile(Sxx_db, 95)
    peaks = (local_max) & (Sxx_db > threshold)

    peak_freq_indices, peak_time_indices = np.where(peaks)
    
    # 3. Generate Hashes
    # INCREASED FAN-OUT: Pairs each peak with the next 15 peaks instead of 5
    fan_out = 15 
    hashes = []
    
    sort_idx = np.argsort(peak_time_indices)
    t_idx_sorted = peak_time_indices[sort_idx]
    f_idx_sorted = peak_freq_indices[sort_idx]
    
    for i in range(len(t_idx_sorted) - fan_out):
        for j in range(1, fan_out + 1):
            t1, f1 = int(t_idx_sorted[i]), int(f_idx_sorted[i])
            t2, f2 = int(t_idx_sorted[i+j]), int(f_idx_sorted[i+j])
            time_delta = t2 - t1
            
            # INCREASED DELTA ALLOWANCE: Allow peaks up to 200 bins apart
            if 0 < time_delta < 200:
                hash_key = f"{f1}_{f2}_{time_delta}"
                hashes.append([hash_key, t1])
                
    return hashes, Sxx_db, t, f, peak_time_indices, peak_freq_indices