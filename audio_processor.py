import librosa
import numpy as np
import soundfile as sf
from typing import Tuple, List
from scipy.stats import skew, kurtosis
import warnings
warnings.filterwarnings('ignore')


class AudioProcessor:
    """
    Audio processing utilities for baby cry analysis.
    Extracts relevant features from audio files for ML classification.
    """
    
    def __init__(self, sample_rate: int = 22050, duration: float = 3.0):
        """
        Initialize audio processor.
        
        Args:
            sample_rate: Target sample rate for audio processing
            duration: Maximum duration to process (in seconds)
        """
        self.sample_rate = sample_rate
        self.duration = duration
        self.max_samples = int(sample_rate * duration)
    
    def load_audio(self, file_path: str) -> Tuple[np.ndarray, int]:
        """
        Load audio file and normalize it.
        
        Args:
            file_path: Path to the audio file
            
        Returns:
            Tuple of (audio_data, sample_rate)
        """
        try:
            # Load audio file
            audio, sr = librosa.load(file_path, sr=self.sample_rate, duration=self.duration)
            
            # Normalize audio
            if len(audio) > 0:
                audio = audio / np.max(np.abs(audio) + 1e-8)
            
            # Pad or truncate to fixed length
            if len(audio) < self.max_samples:
                audio = np.pad(audio, (0, self.max_samples - len(audio)))
            else:
                audio = audio[:self.max_samples]
                
            return audio, sr
        except Exception as e:
            raise Exception(f"Error loading audio file {file_path}: {str(e)}")
    
    def extract_mfcc_features(self, audio: np.ndarray, n_mfcc: int = 13) -> np.ndarray:
        """
        Extract MFCC (Mel-Frequency Cepstral Coefficients) features.
        
        Args:
            audio: Audio signal
            n_mfcc: Number of MFCC coefficients
            
        Returns:
            MFCC features
        """
        mfccs = librosa.feature.mfcc(y=audio, sr=self.sample_rate, n_mfcc=n_mfcc)
        return mfccs
    
    def extract_spectral_features(self, audio: np.ndarray) -> dict:
        """
        Extract spectral features from audio.
        
        Args:
            audio: Audio signal
            
        Returns:
            Dictionary of spectral features
        """
        # Spectral centroid
        spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=self.sample_rate)[0]
        
        # Spectral rolloff
        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=self.sample_rate)[0]
        
        # Spectral bandwidth
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=self.sample_rate)[0]
        
        # Zero crossing rate
        zcr = librosa.feature.zero_crossing_rate(audio)[0]
        
        return {
            'spectral_centroid': spectral_centroid,
            'spectral_rolloff': spectral_rolloff,
            'spectral_bandwidth': spectral_bandwidth,
            'zero_crossing_rate': zcr
        }
    
    def extract_chroma_features(self, audio: np.ndarray) -> np.ndarray:
        """
        Extract chroma features.
        
        Args:
            audio: Audio signal
            
        Returns:
            Chroma features
        """
        chroma = librosa.feature.chroma_stft(y=audio, sr=self.sample_rate)
        return chroma
    
    def extract_tempo_and_beat(self, audio: np.ndarray) -> Tuple[float, np.ndarray]:
        """
        Extract tempo and beat features.
        
        Args:
            audio: Audio signal
            
        Returns:
            Tuple of (tempo, beat_frames)
        """
        tempo, beat_frames = librosa.beat.beat_track(y=audio, sr=self.sample_rate)
        return tempo, beat_frames
    
    def extract_statistical_features(self, features: np.ndarray) -> np.ndarray:
        """
        Extract statistical features from a feature matrix.
        
        Args:
            features: Feature matrix (features x time)
            
        Returns:
            Statistical features array
        """
        stats = []
        for feature_row in features:
            stats.extend([
                np.mean(feature_row),
                np.std(feature_row),
                np.min(feature_row),
                np.max(feature_row),
                skew(feature_row),
                kurtosis(feature_row)
            ])
        return np.array(stats)
    
    def extract_all_features(self, audio: np.ndarray) -> np.ndarray:
        """
        Extract comprehensive feature set from audio.
        
        Args:
            audio: Audio signal
            
        Returns:
            Feature vector
        """
        features = []
        
        # MFCC features
        mfccs = self.extract_mfcc_features(audio)
        mfcc_stats = self.extract_statistical_features(mfccs)
        features.extend(mfcc_stats)
        
        # Spectral features
        spectral_features = self.extract_spectral_features(audio)
        for feature_name, feature_values in spectral_features.items():
            feature_stats = [
                np.mean(feature_values),
                np.std(feature_values),
                np.min(feature_values),
                np.max(feature_values)
            ]
            features.extend(feature_stats)
        
        # Chroma features
        chroma = self.extract_chroma_features(audio)
        chroma_stats = self.extract_statistical_features(chroma)
        features.extend(chroma_stats)
        
        # Tempo
        tempo, _ = self.extract_tempo_and_beat(audio)
        features.append(tempo)
        
        # Energy and power features
        energy = np.sum(audio ** 2)
        power = np.mean(audio ** 2)
        features.extend([energy, power])
        
        return np.array(features)
    
    def process_file(self, file_path: str) -> np.ndarray:
        """
        Process an audio file and extract features.
        
        Args:
            file_path: Path to the audio file
            
        Returns:
            Feature vector
        """
        audio, _ = self.load_audio(file_path)
        features = self.extract_all_features(audio)
        return features
    
    def save_audio(self, audio: np.ndarray, file_path: str):
        """
        Save audio to file.
        
        Args:
            audio: Audio signal
            file_path: Output file path
        """
        sf.write(file_path, audio, self.sample_rate)
