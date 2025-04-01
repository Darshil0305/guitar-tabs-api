import os
import logging
from pathlib import Path
import numpy as np
from pydub import AudioSegment
import tempfile
import shutil

logger = logging.getLogger(__name__)

# Try to import Spleeter, but don't fail if it's not available
SPLEETER_AVAILABLE = False
try:
    from spleeter.separator import Separator
    SPLEETER_AVAILABLE = True
    logger.info("Spleeter is available")
except ImportError:
    logger.warning("Spleeter is not available. Source separation will be disabled.")

class SourceSeparation:
    """Service for isolating guitar tracks from mixed audio using Spleeter."""
    
    def __init__(self, temp_dir=None):
        """Initialize the source separation service.
        
        Args:
            temp_dir: Directory to store temporary files.
        """
        self.temp_dir = temp_dir or tempfile.mkdtemp()
        self.separator = None
        
    def initialize_separator(self):
        """Initialize the Spleeter separator with the 4stems model.
        
        The 4stems model separates audio into vocals, drums, bass, and other.
        The "other" stem typically contains guitars and other melodic instruments.
        """
        if not SPLEETER_AVAILABLE:
            logger.warning("Spleeter is not available. Source separation is disabled.")
            return False
            
        if self.separator is None:
            try:
                self.separator = Separator('spleeter:4stems')
                logger.info("Spleeter separator initialized successfully")
                return True
            except Exception as e:
                logger.error(f"Failed to initialize Spleeter separator: {e}")
                raise
        return True
    
    def isolate_guitar(self, audio_path):
        """Isolate guitar track from the input audio file.
        
        Args:
            audio_path: Path to the input audio file.
            
        Returns:
            Path to the isolated guitar track audio file or original audio path if separation is not available.
        """
        # Check if Spleeter is available
        if not SPLEETER_AVAILABLE:
            logger.warning("Source separation not available. Using original audio.")
            return audio_path
            
        # Try to initialize separator
        if not self.initialize_separator():
            logger.warning("Failed to initialize separator. Using original audio.")
            return audio_path
        
        input_path = Path(audio_path)
        if not input_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        output_dir = os.path.join(self.temp_dir, "separated")
        
        try:
            # Separate the audio into stems
            self.separator.separate_to_file(
                str(input_path),
                output_dir,
                filename_format='{instrument}.{codec}'
            )
            logger.info(f"Successfully separated audio: {audio_path}")
            
            # The "other" stem contains guitars and other melodic instruments
            isolated_path = os.path.join(output_dir, "other.wav")
            
            if not os.path.exists(isolated_path):
                raise FileNotFoundError(f"Isolated guitar track not found: {isolated_path}")
            
            # Create a new file path for the enhanced guitar track
            enhanced_path = os.path.join(self.temp_dir, "guitar_isolated.wav")
            
            # Apply some post-processing to enhance the guitar signal
            self._enhance_guitar_track(isolated_path, enhanced_path)
            
            return enhanced_path
            
        except Exception as e:
            logger.error(f"Error in source separation: {e}")
            # Fall back to the original audio if separation fails
            return audio_path
    
    def _enhance_guitar_track(self, input_path, output_path):
        """Apply post-processing to enhance the isolated guitar track.
        
        Args:
            input_path: Path to the isolated "other" stem.
            output_path: Path to save the enhanced guitar track.
        """
        try:
            # Load the audio file
            audio = AudioSegment.from_wav(input_path)
            
            # Apply some basic enhancements
            # 1. Normalize the volume
            audio = audio.normalize()
            
            # 2. Apply mild compression to even out the dynamics
            # This is a simplified approach - in a real implementation,
            # we'd use a proper compressor
            audio = audio.compress_dynamic_range()
            
            # 3. Enhance mid-range frequencies where guitar typically resides
            # In a full implementation, we'd use more sophisticated EQ techniques
            
            # Save the enhanced audio
            audio.export(output_path, format="wav")
            logger.info(f"Enhanced guitar track saved to: {output_path}")
            
        except Exception as e:
            logger.error(f"Error enhancing guitar track: {e}")
            # Fall back to the original isolated track
            shutil.copy(input_path, output_path)
    
    def cleanup(self):
        """Clean up temporary files."""
        try:
            if os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir)
                logger.info(f"Cleaned up temporary directory: {self.temp_dir}")
        except Exception as e:
            logger.error(f"Error cleaning up temporary files: {e}") 