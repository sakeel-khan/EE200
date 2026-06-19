import os
import json
import glob
from audio_processor import extract_fingerprint

# Set this to the folder where you extracted your zip file of mp3s
SONG_FOLDER = "songs_folder" 
DATABASE_FILE = "database.json"

def build_db():
    print("Starting database generation...")
    database = {}
    
    # Find all mp3 files in the specified folder
    search_pattern = os.path.join(SONG_FOLDER, "*.mp3")
    song_files = glob.glob(search_pattern)
    
    if not song_files:
        print(f"No .mp3 files found in {SONG_FOLDER}!")
        return

    for file_path in song_files:
        # Extract filename without extension (e.g., "song1")
        song_name = os.path.splitext(os.path.basename(file_path))[0]
        print(f"Processing {song_name}...")
        
        hashes, _, _, _, _, _ = extract_fingerprint(file_path)
        
        if hashes:
            database[song_name] = hashes
            print(f" -> Generated {len(hashes)} hashes.")
            
    # Save to JSON
    with open(DATABASE_FILE, "w") as f:
        json.dump(database, f)
        
    print(f"Database successfully saved to {DATABASE_FILE} with {len(database)} songs.")

if __name__ == "__main__":
    # Create the folder if it doesn't exist so you can drop your mp3s in it
    if not os.path.exists(SONG_FOLDER):
        os.makedirs(SONG_FOLDER)
        print(f"Created folder '{SONG_FOLDER}'. Please put your .mp3 files inside and run this again.")
    else:
        build_db()