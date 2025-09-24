"""
Audio processing utilities for ACE-Steps BentoML Service
"""

import os
import logging
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
import torch
import torchaudio
import numpy as np
import librosa
import soundfile as sf

logger = logging.getLogger(__name__)

class AudioProcessor:
    """
    Audio processing utilities for the ACE-Steps service
    """
    
    def __init__(self):
        self.supported_formats = ['wav', 'mp3', 'flac', 'ogg']
        self.default_sample_rate = 44100
        
    def validate_audio_file(self, file_path: str) -> bool:
        """
        Validate if the audio file is readable and supported
        
        Args:
            file_path: Path to the audio file
            
        Returns:
            bool: True if valid, False otherwise
        """
        try:
            if not os.path.exists(file_path):
                return False
                
            # Try to load the audio file
            audio, sr = librosa.load(file_path, sr=None)
            return len(audio) > 0
            
        except Exception as e:
            logger.warning(f"Audio file validation failed: {str(e)}")
            return False
    
    def get_audio_info(self, file_path: str) -> Dict[str, Any]:
        """
        Get audio file information
        
        Args:
            file_path: Path to the audio file
            
        Returns:
            Dict[str, Any]: Audio information
        """
        try:
            if not os.path.exists(file_path):
                return {"error": "File not found"}
            
            # Load audio with librosa
            audio, sr = librosa.load(file_path, sr=None)
            
            # Get file size
            file_size = os.path.getsize(file_path)
            
            # Calculate duration
            duration = len(audio) / sr
            
            # Get format from extension
            file_format = Path(file_path).suffix.lower().lstrip('.')
            
            return {
                "file_path": file_path,
                "sample_rate": sr,
                "duration": duration,
                "channels": 1 if audio.ndim == 1 else audio.shape[0],
                "file_size": file_size,
                "format": file_format,
                "samples": len(audio)
            }
            
        except Exception as e:
            logger.error(f"Failed to get audio info: {str(e)}")
            return {"error": str(e)}
    
    def convert_audio_format(
        self, 
        input_path: str, 
        output_path: str, 
        target_format: str = "wav",
        target_sr: Optional[int] = None
    ) -> bool:
        """
        Convert audio file to target format
        
        Args:
            input_path: Input audio file path
            output_path: Output audio file path
            target_format: Target audio format
            target_sr: Target sample rate (None to keep original)
            
        Returns:
            bool: True if conversion successful, False otherwise
        """
        try:
            # Load audio
            audio, sr = librosa.load(input_path, sr=target_sr)
            
            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Save in target format
            if target_format.lower() == 'wav':
                sf.write(output_path, audio, sr)
            elif target_format.lower() == 'mp3':
                # Use torchaudio for MP3
                audio_tensor = torch.from_numpy(audio).float()
                torchaudio.save(output_path, audio_tensor.unsqueeze(0), sr)
            else:
                # Use soundfile for other formats
                sf.write(output_path, audio, sr)
            
            logger.info(f"Audio converted: {input_path} -> {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Audio conversion failed: {str(e)}")
            return False
    
    def normalize_audio(self, audio: np.ndarray, target_db: float = -20.0) -> np.ndarray:
        """
        Normalize audio to target dB level
        
        Args:
            audio: Audio array
            target_db: Target dB level
            
        Returns:
            np.ndarray: Normalized audio
        """
        try:
            # Calculate RMS
            rms = np.sqrt(np.mean(audio**2))
            
            if rms > 0:
                # Calculate target RMS
                target_rms = 10**(target_db / 20)
                
                # Normalize
                normalized_audio = audio * (target_rms / rms)
                
                # Clip to prevent clipping
                normalized_audio = np.clip(normalized_audio, -1.0, 1.0)
                
                return normalized_audio
            else:
                return audio
                
        except Exception as e:
            logger.error(f"Audio normalization failed: {str(e)}")
            return audio
    
    def trim_silence(
        self, 
        audio: np.ndarray, 
        sr: int, 
        top_db: float = 20.0,
        frame_length: int = 2048,
        hop_length: int = 512
    ) -> np.ndarray:
        """
        Trim silence from audio
        
        Args:
            audio: Audio array
            sr: Sample rate
            top_db: Silence threshold in dB
            frame_length: Frame length for analysis
            hop_length: Hop length for analysis
            
        Returns:
            np.ndarray: Trimmed audio
        """
        try:
            # Use librosa to trim silence
            trimmed_audio, _ = librosa.effects.trim(
                audio, 
                top_db=top_db,
                frame_length=frame_length,
                hop_length=hop_length
            )
            
            return trimmed_audio
            
        except Exception as e:
            logger.error(f"Silence trimming failed: {str(e)}")
            return audio
    
    def resample_audio(
        self, 
        audio: np.ndarray, 
        original_sr: int, 
        target_sr: int
    ) -> Tuple[np.ndarray, int]:
        """
        Resample audio to target sample rate
        
        Args:
            audio: Audio array
            original_sr: Original sample rate
            target_sr: Target sample rate
            
        Returns:
            Tuple[np.ndarray, int]: Resampled audio and sample rate
        """
        try:
            if original_sr == target_sr:
                return audio, target_sr
            
            # Use librosa for resampling
            resampled_audio = librosa.resample(
                audio, 
                orig_sr=original_sr, 
                target_sr=target_sr
            )
            
            return resampled_audio, target_sr
            
        except Exception as e:
            logger.error(f"Audio resampling failed: {str(e)}")
            return audio, original_sr
    
    def get_audio_spectrum(self, file_path: str) -> Dict[str, Any]:
        """
        Get audio spectrum information
        
        Args:
            file_path: Path to audio file
            
        Returns:
            Dict[str, Any]: Spectrum information
        """
        try:
            # Load audio
            audio, sr = librosa.load(file_path, sr=None)
            
            # Compute spectral features
            spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
            spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)[0]
            zero_crossing_rate = librosa.feature.zero_crossing_rate(audio)[0]
            
            # Compute MFCCs
            mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
            
            return {
                "spectral_centroid_mean": float(np.mean(spectral_centroids)),
                "spectral_centroid_std": float(np.std(spectral_centroids)),
                "spectral_rolloff_mean": float(np.mean(spectral_rolloff)),
                "spectral_rolloff_std": float(np.std(spectral_rolloff)),
                "zero_crossing_rate_mean": float(np.mean(zero_crossing_rate)),
                "zero_crossing_rate_std": float(np.std(zero_crossing_rate)),
                "mfcc_mean": [float(x) for x in np.mean(mfccs, axis=1)],
                "mfcc_std": [float(x) for x in np.std(mfccs, axis=1)]
            }
            
        except Exception as e:
            logger.error(f"Spectrum analysis failed: {str(e)}")
            return {"error": str(e)}
    
    def cleanup_old_files(self, directory: str, max_age_hours: int = 24) -> int:
        """
        Clean up old audio files
        
        Args:
            directory: Directory to clean up
            max_age_hours: Maximum age in hours
            
        Returns:
            int: Number of files cleaned up
        """
        try:
            if not os.path.exists(directory):
                return 0
            
            import time
            current_time = time.time()
            max_age_seconds = max_age_hours * 3600
            cleaned_count = 0
            
            for filename in os.listdir(directory):
                file_path = os.path.join(directory, filename)
                
                if os.path.isfile(file_path):
                    file_age = current_time - os.path.getmtime(file_path)
                    
                    if file_age > max_age_seconds:
                        try:
                            os.remove(file_path)
                            cleaned_count += 1
                            logger.info(f"Cleaned up old file: {file_path}")
                        except Exception as e:
                            logger.warning(f"Failed to remove file {file_path}: {str(e)}")
            
            return cleaned_count
            
        except Exception as e:
            logger.error(f"File cleanup failed: {str(e)}")
            return 0
