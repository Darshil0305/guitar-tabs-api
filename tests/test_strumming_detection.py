import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from app.services.tab_generator import (
    analyze_rhythm, 
    detect_beat_emphasis, 
    detect_strumming_pattern,
    STRUMMING_PATTERNS
)

class TestStrummingDetection:
    """Tests for the strumming pattern detection functionality."""
    
    def test_analyze_rhythm_with_data(self):
        """Test rhythm analysis with valid data."""
        # Arrange
        y = np.random.random(22050 * 5)  # 5 seconds of random audio
        sr = 22050  # Sample rate
        onset_times = np.linspace(0, 4.5, 20)  # 20 evenly spaced onsets
        
        # Act
        result = analyze_rhythm(y, sr, onset_times)
        
        # Assert
        assert isinstance(result, dict)
        assert 'tempo' in result
        assert 'mean_ioi' in result
        assert 'rhythm_consistency' in result
        assert 'beat_emphasis' in result
        assert 'onset_count' in result
        assert result['onset_count'] == len(onset_times)
        assert result['mean_ioi'] > 0
    
    def test_analyze_rhythm_empty_onsets(self):
        """Test rhythm analysis with no onsets."""
        # Arrange
        y = np.random.random(22050 * 5)
        sr = 22050
        onset_times = []
        
        # Act
        result = analyze_rhythm(y, sr, onset_times)
        
        # Assert
        assert result['onset_count'] == 0
        assert result['mean_ioi'] == 0
        assert result['rhythm_consistency'] == 1.0  # Default value
    
    def test_analyze_rhythm_single_onset(self):
        """Test rhythm analysis with a single onset."""
        # Arrange
        y = np.random.random(22050 * 5)
        sr = 22050
        onset_times = [1.0]  # Single onset at 1 second
        
        # Act
        result = analyze_rhythm(y, sr, onset_times)
        
        # Assert
        assert result['onset_count'] == 1
        assert result['mean_ioi'] == 0  # No intervals with one onset
    
    @patch('app.services.tab_generator.librosa.beat.beat_track')
    def test_detect_beat_emphasis_all_on_beats(self, mock_beat_track):
        """Test detection when all onsets are on the beats."""
        # Arrange
        beat_times = np.array([1.0, 2.0, 3.0, 4.0])
        onset_times = np.array([0.98, 1.97, 3.02, 4.05])  # Very close to beats
        
        # Act
        emphasis = detect_beat_emphasis(onset_times, beat_times)
        
        # Assert
        assert emphasis > 0.8  # Strong emphasis on beats
    
    @patch('app.services.tab_generator.librosa.beat.beat_track')
    def test_detect_beat_emphasis_all_offbeats(self, mock_beat_track):
        """Test detection when all onsets are on the offbeats."""
        # Arrange
        beat_times = np.array([1.0, 2.0, 3.0, 4.0])
        # Halfway between beats
        onset_times = np.array([1.5, 2.5, 3.5])
        
        # Act
        emphasis = detect_beat_emphasis(onset_times, beat_times)
        
        # Assert
        assert emphasis < -0.2  # Emphasis on offbeats
    
    @patch('app.services.tab_generator.librosa.beat.beat_track')
    def test_detect_beat_emphasis_mixed(self, mock_beat_track):
        """Test detection with mixed on/offbeats."""
        # Arrange
        beat_times = np.array([1.0, 2.0, 3.0, 4.0])
        # Mixture of on and offbeats
        onset_times = np.array([1.02, 1.5, 2.05, 2.5, 3.01, 3.5])
        
        # Act
        emphasis = detect_beat_emphasis(onset_times, beat_times)
        
        # Assert
        assert -0.2 <= emphasis <= 0.2  # Neutral emphasis
    
    def test_detect_beat_emphasis_empty_inputs(self):
        """Test with empty inputs."""
        # Arrange
        beat_times = np.array([])
        onset_times = np.array([])
        
        # Act
        emphasis = detect_beat_emphasis(onset_times, beat_times)
        
        # Assert
        assert emphasis == 0.0  # Neutral when no data
    
    def test_detect_strumming_pattern_fingerstyle(self):
        """Test strumming pattern detection for fingerstyle."""
        # Arrange
        rhythm_features = {
            'tempo': 120,
            'rhythm_consistency': 0.3,
            'beat_emphasis': 0.5,
            'mean_ioi': 0.25,
            'onset_count': 20
        }
        
        # Act
        pattern = detect_strumming_pattern(rhythm_features, is_fingerstyle=True)
        
        # Assert
        assert pattern == "N/A (Fingerstyle)"
    
    def test_detect_strumming_pattern_variable_rhythm(self):
        """Test strumming pattern detection with variable rhythm."""
        # Arrange
        rhythm_features = {
            'tempo': 120,
            'rhythm_consistency': 0.7,  # High value = variable
            'beat_emphasis': 0.4,
            'mean_ioi': 0.25,
            'onset_count': 20
        }
        
        # Act
        pattern = detect_strumming_pattern(rhythm_features, is_fingerstyle=False)
        
        # Assert
        assert pattern == STRUMMING_PATTERNS['ballad']
    
    def test_detect_strumming_pattern_consistent_downbeats(self):
        """Test strumming pattern detection with consistent downbeats."""
        # Arrange
        rhythm_features = {
            'tempo': 90,
            'rhythm_consistency': 0.2,  # Low value = consistent
            'beat_emphasis': 0.8,  # High value = emphasis on beats
            'mean_ioi': 0.25,
            'onset_count': 20
        }
        
        # Act
        pattern = detect_strumming_pattern(rhythm_features, is_fingerstyle=False)
        
        # Assert
        assert pattern == STRUMMING_PATTERNS['waltz']
    
    def test_detect_strumming_pattern_fast_consistent_downbeats(self):
        """Test strumming pattern detection with fast, consistent downbeats."""
        # Arrange
        rhythm_features = {
            'tempo': 140,
            'rhythm_consistency': 0.2,
            'beat_emphasis': 0.8,
            'mean_ioi': 0.2,
            'onset_count': 20
        }
        
        # Act
        pattern = detect_strumming_pattern(rhythm_features, is_fingerstyle=False)
        
        # Assert
        assert pattern == STRUMMING_PATTERNS['basic']
    
    def test_detect_strumming_pattern_offbeats(self):
        """Test strumming pattern detection with emphasis on offbeats."""
        # Arrange
        rhythm_features = {
            'tempo': 100,
            'rhythm_consistency': 0.3,
            'beat_emphasis': -0.5,  # Negative value = offbeats emphasis
            'mean_ioi': 0.25,
            'onset_count': 20
        }
        
        # Act
        pattern = detect_strumming_pattern(rhythm_features, is_fingerstyle=False)
        
        # Assert
        assert pattern == STRUMMING_PATTERNS['reggae'] 