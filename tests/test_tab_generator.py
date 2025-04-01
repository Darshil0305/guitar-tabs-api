import unittest
from unittest.mock import patch, MagicMock
import os
import sys
import tempfile

# Add the app directory to the path so we can import from it
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.services.tab_generator import (
    extract_video_id,
    pitch_to_midi_note,
    midi_to_note_name,
    note_to_tab_position,
    generate_tab_notation
)

class TestTabGenerator(unittest.TestCase):
    """Tests for the tab generator module"""
    
    def test_extract_video_id(self):
        """Test extracting video ID from different YouTube URL formats"""
        # Standard format
        self.assertEqual(
            extract_video_id('https://www.youtube.com/watch?v=dQw4w9WgXcQ'),
            'dQw4w9WgXcQ'
        )
        
        # Short format
        self.assertEqual(
            extract_video_id('https://youtu.be/dQw4w9WgXcQ'),
            'dQw4w9WgXcQ'
        )
        
        # With additional parameters
        self.assertEqual(
            extract_video_id('https://www.youtube.com/watch?v=dQw4w9WgXcQ&t=42s'),
            'dQw4w9WgXcQ'
        )
        
        # Invalid URL
        self.assertIsNone(extract_video_id('https://www.example.com'))
    
    def test_pitch_to_midi_note(self):
        """Test converting pitch frequency to MIDI note number"""
        # A4 = 440Hz = MIDI note 69
        self.assertEqual(pitch_to_midi_note(440.0), 69)
        
        # C4 = 261.63Hz ≈ MIDI note 60
        self.assertEqual(pitch_to_midi_note(261.63), 60)
        
        # Invalid pitch
        self.assertEqual(pitch_to_midi_note(0), 0)
    
    def test_midi_to_note_name(self):
        """Test converting MIDI note numbers to note names"""
        # MIDI note 60 = C4
        self.assertEqual(midi_to_note_name(60), 'C4')
        
        # MIDI note 69 = A4
        self.assertEqual(midi_to_note_name(69), 'A4')
        
        # Invalid MIDI note
        self.assertEqual(midi_to_note_name(0), '')
    
    def test_note_to_tab_position(self):
        """Test converting MIDI notes to guitar tab positions"""
        # Open low E string = MIDI note 40
        self.assertEqual(note_to_tab_position(40), (5, 0))
        
        # 5th fret on A string = MIDI note 50 (D3)
        self.assertEqual(note_to_tab_position(50), (3, 0))
        
        # Note too low for standard tuning
        self.assertIsNone(note_to_tab_position(35))
    
    def test_generate_tab_notation(self):
        """Test generating tab notation from notes"""
        # Simple example with a few notes
        notes = [
            (0.0, 440.0),  # A4 (5th fret on high E string)
            (0.5, 392.0),  # G4 (3rd fret on high E string)
            (1.0, 329.63)  # E4 (open high E string)
        ]
        
        tab = generate_tab_notation(notes)
        
        # Check if all strings are present in the tab
        self.assertIn('E|', tab)
        self.assertIn('B|', tab)
        self.assertIn('G|', tab)
        self.assertIn('D|', tab)
        self.assertIn('A|', tab)
        self.assertIn('E|', tab)
        
        # Check for strumming pattern text
        self.assertIn('Strumming pattern', tab)
        
        # Test with capo
        tab_capo = generate_tab_notation(notes, use_capo=True)
        # We can't easily predict the capo position, but we can check that it contains the capo text
        # if a capo is recommended
        if 'Capo on fret' in tab_capo:
            self.assertRegex(tab_capo, r'Capo on fret \d+')
        
        # Test with fingerstyle
        tab_fingerstyle = generate_tab_notation(notes, is_fingerstyle=True)
        self.assertIn('Fingerstyle pattern', tab_fingerstyle)

if __name__ == '__main__':
    unittest.main() 