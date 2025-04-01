import os
import pytest
import tempfile
import shutil
from unittest.mock import patch, MagicMock
from app.services.source_separation import SourceSeparation

class TestSourceSeparation:
    """Test cases for the SourceSeparation service."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for testing."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        # Clean up after tests
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def sample_audio_file(self, temp_dir):
        """Create a sample audio file for testing."""
        audio_path = os.path.join(temp_dir, "sample.wav")
        # Create an empty file
        with open(audio_path, 'w') as f:
            f.write("test")
        return audio_path
    
    @patch('app.services.source_separation.Separator')
    def test_initialize_separator(self, mock_separator):
        """Test that the separator is initialized correctly."""
        # Arrange
        service = SourceSeparation()
        mock_separator.return_value = MagicMock()
        
        # Act
        service.initialize_separator()
        
        # Assert
        mock_separator.assert_called_once_with('spleeter:4stems')
        assert service.separator is not None
    
    @patch('app.services.source_separation.Separator')
    def test_isolate_guitar_success(self, mock_separator, sample_audio_file, temp_dir):
        """Test successful guitar isolation."""
        # Arrange
        service = SourceSeparation(temp_dir=temp_dir)
        mock_separator_instance = MagicMock()
        mock_separator.return_value = mock_separator_instance
        
        # Create the expected output directory and file
        output_dir = os.path.join(temp_dir, "separated")
        os.makedirs(output_dir, exist_ok=True)
        isolated_path = os.path.join(output_dir, "other.wav")
        with open(isolated_path, 'w') as f:
            f.write("isolated audio")
        
        # Act
        with patch('app.services.source_separation.AudioSegment') as mock_audio_segment:
            mock_audio = MagicMock()
            mock_audio_segment.from_wav.return_value = mock_audio
            mock_audio.normalize.return_value = mock_audio
            mock_audio.compress_dynamic_range.return_value = mock_audio
            
            result = service.isolate_guitar(sample_audio_file)
        
        # Assert
        assert result == os.path.join(temp_dir, "guitar_isolated.wav")
        mock_separator_instance.separate_to_file.assert_called_once()
        mock_audio_segment.from_wav.assert_called_once_with(isolated_path)
        mock_audio.export.assert_called_once()
    
    def test_isolate_guitar_file_not_found(self, temp_dir):
        """Test error handling when input file doesn't exist."""
        # Arrange
        service = SourceSeparation(temp_dir=temp_dir)
        non_existent_file = os.path.join(temp_dir, "nonexistent.wav")
        
        # Act & Assert
        with pytest.raises(FileNotFoundError):
            service.isolate_guitar(non_existent_file)
    
    @patch('app.services.source_separation.Separator')
    def test_cleanup(self, mock_separator, temp_dir):
        """Test that temporary files are cleaned up."""
        # Arrange
        service = SourceSeparation(temp_dir=temp_dir)
        
        # Create some test files in the temp directory
        test_file = os.path.join(temp_dir, "test.txt")
        with open(test_file, 'w') as f:
            f.write("test data")
        
        # Act
        service.cleanup()
        
        # Assert
        assert not os.path.exists(temp_dir)
    
    @patch('app.services.source_separation.Separator')
    def test_enhance_guitar_track(self, mock_separator, temp_dir, sample_audio_file):
        """Test the guitar track enhancement function."""
        # Arrange
        service = SourceSeparation(temp_dir=temp_dir)
        output_path = os.path.join(temp_dir, "enhanced.wav")
        
        # Act
        with patch('app.services.source_separation.AudioSegment') as mock_audio_segment:
            mock_audio = MagicMock()
            mock_audio_segment.from_wav.return_value = mock_audio
            mock_audio.normalize.return_value = mock_audio
            mock_audio.compress_dynamic_range.return_value = mock_audio
            
            service._enhance_guitar_track(sample_audio_file, output_path)
        
        # Assert
        mock_audio_segment.from_wav.assert_called_once_with(sample_audio_file)
        mock_audio.normalize.assert_called_once()
        mock_audio.compress_dynamic_range.assert_called_once()
        mock_audio.export.assert_called_once_with(output_path, format="wav") 