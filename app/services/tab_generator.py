import os
import tempfile
import yt_dlp
import librosa
import numpy as np
from pydub import AudioSegment
import re

# Constants for musical notes and guitar strings
NOTES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
GUITAR_STRINGS = ['E', 'B', 'G', 'D', 'A', 'E']  # High to low
GUITAR_TUNING = [64, 59, 55, 50, 45, 40]  # MIDI note numbers for standard tuning

# Directory for temporary files
TEMP_DIR = tempfile.gettempdir()

def extract_video_id(url):
    """Extract YouTube video ID from URL"""
    # Handle standard YouTube URL format
    standard_regex = re.compile(r'(?:youtube\.com\/watch\?v=|youtu\.be\/)([a-zA-Z0-9_-]{11})')
    match = standard_regex.search(url)
    
    return match.group(1) if match else None

def download_audio(youtube_url):
    """Download audio from YouTube URL using yt-dlp"""
    video_id = extract_video_id(youtube_url)
    if not video_id:
        raise ValueError("Invalid YouTube URL")
    
    output_path = os.path.join(TEMP_DIR, f"{video_id}.mp3")
    
    # Check if file already exists
    if os.path.exists(output_path):
        return output_path
    
    # yt-dlp options
    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'outtmpl': os.path.join(TEMP_DIR, f"{video_id}.%(ext)s"),
        'quiet': True,
        'no_warnings': True
    }
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([youtube_url])
        return output_path
    except Exception as e:
        raise Exception(f"Error downloading YouTube audio: {str(e)}")

def analyze_audio(audio_path):
    """
    Analyze audio file to detect pitches and onsets
    Returns a list of (time, note) tuples
    """
    # Load audio file with librosa
    y, sr = librosa.load(audio_path, sr=22050)
    
    # Trim silence
    y, _ = librosa.effects.trim(y)
    
    # Detect onsets (note start times)
    onset_frames = librosa.onset.onset_detect(y=y, sr=sr, units='frames')
    onset_times = librosa.frames_to_time(onset_frames, sr=sr)
    
    # Extract melody using pitch detection
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    
    # Extract the most prominent pitch at each onset
    notes = []
    for onset_time, onset_frame in zip(onset_times, onset_frames):
        if onset_frame < pitches.shape[1]:
            index = magnitudes[:, onset_frame].argmax()
            pitch = pitches[index, onset_frame]
            if pitch > 0:  # Ignore silent frames
                notes.append((onset_time, pitch))
    
    return notes

def pitch_to_midi_note(pitch):
    """Convert frequency to MIDI note number"""
    if pitch <= 0:
        return 0
    return int(round(69 + 12 * np.log2(pitch / 440.0)))

def midi_to_note_name(midi_note):
    """Convert MIDI note number to note name"""
    if midi_note <= 0:
        return ""
    note_idx = (midi_note - 21) % 12
    octave = (midi_note - 12) // 12
    return f"{NOTES[note_idx]}{octave}"

def note_to_tab_position(midi_note):
    """
    Convert MIDI note to guitar tab position
    Returns (string_idx, fret) or None if note can't be played
    """
    for i, base_note in enumerate(GUITAR_TUNING):
        fret = midi_note - base_note
        if 0 <= fret <= 24:  # Most guitars have 24 frets
            return (i, fret)
    # If note can't be played on guitar in standard tuning
    return None

def generate_tab_notation(notes, use_capo=False, is_fingerstyle=False):
    """
    Generate guitar tab notation from notes
    Returns a formatted tab string
    """
    # Initialize empty tab
    tab_lines = ["E|", "B|", "G|", "D|", "A|", "E|"]
    
    # Determine if we need capo and at which fret
    capo_fret = 0
    if use_capo:
        # Simple heuristic: find most common fret position and use capo there
        all_frets = []
        for _, pitch in notes:
            midi_note = pitch_to_midi_note(pitch)
            tab_pos = note_to_tab_position(midi_note)
            if tab_pos:
                all_frets.append(tab_pos[1])
        
        if all_frets:
            # Count frequency of each fret number
            fret_counts = {}
            for fret in all_frets:
                if fret > 0:  # Only consider non-open strings
                    fret_counts[fret] = fret_counts.get(fret, 0) + 1
            
            # Find most common fret if any
            if fret_counts:
                capo_fret = max(fret_counts.items(), key=lambda x: x[1])[0]
                # Only use capo if it makes sense (fret 1-5 typically)
                if not (0 < capo_fret <= 5):
                    capo_fret = 0
    
    # Generate tab based on notes
    last_time = -1
    spacing = 2  # Default spacing between notes
    
    for time, pitch in notes:
        # Add spacing based on time difference
        if last_time >= 0:
            time_diff = time - last_time
            # Add extra spacing for longer pauses
            extra_spaces = max(0, int(time_diff * 10) - 1)
            for i in range(6):
                tab_lines[i] += "-" * extra_spaces
        
        midi_note = pitch_to_midi_note(pitch)
        
        # Adjust for capo if needed
        if capo_fret > 0:
            midi_note -= capo_fret
        
        tab_pos = note_to_tab_position(midi_note)
        
        if tab_pos:
            string_idx, fret = tab_pos
            # Add the note to the tab
            for i in range(6):
                if i == string_idx:
                    # Add the fret number with padding for double digits
                    fret_str = str(fret)
                    tab_lines[i] += fret_str + "-" * (spacing - len(fret_str))
                else:
                    if is_fingerstyle:
                        # For fingerstyle, only show notes on their strings
                        tab_lines[i] += "-" * spacing
                    else:
                        # For strumming, indicate strummed strings
                        # Simplistic approach: assume strings 0-3 (high E to D) for up strums,
                        # strings 3-5 (D to low E) for down strums
                        if (i >= 3 and string_idx >= 3) or (i < 3 and string_idx < 3):
                            tab_lines[i] += "x" + "-" * (spacing - 1)
                        else:
                            tab_lines[i] += "-" * spacing
        else:
            # If note can't be played, just add spacing
            for i in range(6):
                tab_lines[i] += "-" * spacing
        
        last_time = time
    
    # Finalize the tab
    for i in range(6):
        tab_lines[i] += "|"
    
    # Add capo information if used
    capo_info = f"\nCapo on fret {capo_fret}" if capo_fret > 0 else ""
    
    # Add playing style information
    style_info = "\nFingerstyle pattern" if is_fingerstyle else "\nStrumming pattern"
    
    # Combine all lines with newlines
    return "\n".join(tab_lines) + capo_info + style_info

def get_song_info(video_id):
    """
    Get song information from YouTube video ID
    In a real implementation, this would use YouTube Data API
    """
    try:
        with yt_dlp.YoutubeDL({'quiet': True}) as ydl:
            info = ydl.extract_info(f"https://www.youtube.com/watch?v={video_id}", download=False)
            title = info.get('title', 'Unknown Song')
            
            # Try to extract artist from title (common format: "Artist - Title")
            artist = 'Unknown Artist'
            if ' - ' in title:
                parts = title.split(' - ', 1)
                artist = parts[0].strip()
                title = parts[1].strip()
            
            return {
                'title': title,
                'artist': artist,
                'video_id': video_id
            }
    except Exception:
        # Return default info if extraction fails
        return {
            'title': 'Unknown Song',
            'artist': 'Unknown Artist',
            'video_id': video_id
        }

def generate_tabs_from_youtube(youtube_url, use_capo=False, is_fingerstyle=False):
    """
    Main function to generate guitar tabs from YouTube URL
    """
    try:
        # Extract video ID
        video_id = extract_video_id(youtube_url)
        if not video_id:
            raise ValueError("Invalid YouTube URL")
        
        # Get song information
        song_info = get_song_info(video_id)
        
        # Download audio
        audio_path = download_audio(youtube_url)
        
        # Analyze audio to detect notes
        detected_notes = analyze_audio(audio_path)
        
        # Generate tab notation
        tab_notation = generate_tab_notation(detected_notes, use_capo, is_fingerstyle)
        
        # Placeholder for strumming pattern (replace with actual detection later)
        strumming_pattern = "D DU UDU (Placeholder)" if not is_fingerstyle else "N/A (Fingerstyle)"
        
        # Combine results
        result = {
            'title': song_info['title'],
            'artist': song_info['artist'],
            'video_id': video_id,
            'tab_content': tab_notation,
            'use_capo': use_capo,
            'is_fingerstyle': is_fingerstyle,
            'strumming_pattern': strumming_pattern  # Add strumming pattern here
        }
        
        return result
    except Exception as e:
        raise Exception(f"Error generating tabs: {str(e)}") 