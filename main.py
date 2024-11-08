
# !pip install numpy pandas astropy scipy scikit-learn torch lightkurve matplotlib loguru

from typing import List, Dict, Optional, Tuple, Union, Any
import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.timeseries import LombScargle
from astropy.stats import sigma_clip
from scipy import stats, signal
from scipy.fft import fft, fftfreq
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from lightkurve import search_lightcurve
import matplotlib.pyplot as plt
from dataclasses import dataclass
import logging
from pathlib import Path
import warnings
from torch.optim import Adam
from astropy.utils.masked import MaskedNDArray
from loguru import logger

warnings.filterwarnings('ignore')


@dataclass
class StarData:
    """Class for holding stellar observation data and metadata."""
    star_id: str
    time: np.ndarray
    flux: np.ndarray
    flux_err: Optional[np.ndarray] = None
    metadata: Optional[Dict] = None

class LightCurveDataset(Dataset):
    """PyTorch Dataset for stellar light curve data."""
    
    def __init__(self, time_series: np.ndarray, sequence_length: int = 50):
        """
        Initialize dataset with time series data.
        
        Args:
            time_series: Array of flux measurements
            sequence_length: Length of sequences for LSTM input
        """
        self.sequence_length = sequence_length
        self.data = torch.FloatTensor(time_series)
        
    def __len__(self) -> int:
        return len(self.data) - self.sequence_length
        
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sequence = self.data[idx:idx + self.sequence_length]
        target = self.data[idx + self.sequence_length]
        return sequence, target

class LSTMPredictor(nn.Module):
    """LSTM model for time series prediction."""
    
    def __init__(self, input_size: int = 1, hidden_size: int = 64, num_layers: int = 2):
        """
        Initialize LSTM model.
        
        Args:
            input_size: Number of input features
            hidden_size: Number of hidden units in LSTM
            num_layers: Number of LSTM layers
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2
        )
        
        self.fc = nn.Linear(hidden_size, 1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        # Initialize hidden state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # LSTM forward pass
        lstm_out, _ = self.lstm(x, (h0, c0))
        
        # Get last output
        last_output = lstm_out[:, -1, :]
        
        # Predict next value
        pred = self.fc(last_output)
        return pred

class StellarConsciousnessAnalyzer:
    """Main class for analyzing potential consciousness-like patterns in stellar data."""
    
    def __init__(self, output_dir: str = "stellar_analysis_results", device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        """
        Initialize the analyzer.
        
        Args:
            output_dir: Directory for saving results
            device: Device to use for PyTorch computations
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.device = device
        logger.info(f"Using device: {self.device}")
        
    def fetch_star_data(self, star_id: str, mission: str = "TESS") -> StarData:
        """
        Fetch light curve data for a given star.
        
        Args:
            star_id: Target star identifier
            mission: Space telescope mission name
        """
        try:
            logger.info(f"Fetching data for star {star_id} from {mission}")
            search_result = search_lightcurve(star_id, mission=mission)
            if len(search_result) == 0:
                raise ValueError(f"No data found for star {star_id}")
                
            lc = search_result[0].download()
            clean_lc = lc.remove_outliers()
            
            return StarData(
                star_id=star_id,
                time=clean_lc.time.value,
                flux=clean_lc.flux.value,
                flux_err=clean_lc.flux_err.value if hasattr(clean_lc, 'flux_err') else None
            )
            
        except Exception as e:
            logger.error(f"Error fetching data: {str(e)}")
            raise
            
    def calculate_entropy(self, data: Union[np.ndarray, MaskedNDArray]) -> float:
          """
          Calculate Shannon entropy of the time series.
          
          Args:
              data: Time series data (can be masked array)
              
          Returns:
              float: Shannon entropy value
          """
          # Convert masked array to regular numpy array, replacing masked values with NaN
          if isinstance(data, MaskedNDArray):
              data = data.filled(np.nan)
          
          # Remove NaN values
          data = data[~np.isnan(data)]
          
          if len(data) == 0:
              logger.warning("No valid data points for entropy calculation")
              return 0.0
          
          # Normalize data to [0, 1] range for better histogram binning
          data_normalized = (data - np.min(data)) / (np.max(data) - np.min(data))
          
          # Calculate histogram with automatic bin selection
          hist, _ = np.histogram(data_normalized, bins='sturges', density=True)
          
          # Remove zero-count bins and normalize
          hist = hist[hist > 0]
          hist = hist / hist.sum()
          
          # Calculate entropy
          entropy = -np.sum(hist * np.log2(hist))
          return entropy

    def perform_frequency_analysis(self, data: Union[np.ndarray, MaskedNDArray], sample_rate: float) -> Dict[str, Any]:
        """
        Perform Fourier analysis on the time series.
        
        Args:
            data: Time series data (can be masked array)
            sample_rate: Sampling rate of the data
        """
        # Convert masked array to regular numpy array
        if isinstance(data, MaskedNDArray):
            data = data.filled(np.nan)
        
        # Interpolate NaN values
        nans = np.isnan(data)
        if np.any(nans):
            x = np.arange(len(data))
            data = np.interp(x, x[~nans], data[~nans])
        
        # Rest of the frequency analysis remains the same...
        fft_vals = fft(data)
        freqs = fftfreq(len(data), 1/sample_rate)
        
        pos_mask = freqs >= 0
        freqs = freqs[pos_mask]
        fft_vals = np.abs(fft_vals[pos_mask])
        
        # Find dominant frequencies
        peak_indices = signal.find_peaks(fft_vals)[0]
        dominant_freqs = freqs[peak_indices]
        
        return {
            "frequencies": freqs,
            "amplitudes": fft_vals,
            "dominant_frequencies": dominant_freqs
        }

        
    def train_lstm_model(self, data: Union[np.ndarray, MaskedNDArray], sequence_length: int = 50,
                        batch_size: int = 32, num_epochs: int = 100) -> Tuple[nn.Module, float]:
        """
        Train LSTM model on the time series data.
        """
        # Convert masked array to regular numpy array
        if isinstance(data, MaskedNDArray):
            data = data.filled(np.nan)
        
        # Handle NaN values by interpolation
        nans = np.isnan(data)
        if np.any(nans):
            x = np.arange(len(data))
            data = np.interp(x, x[~nans], data[~nans])
        
        # Rest of the training code remains the same...
        scaler = StandardScaler()
        normalized_data = scaler.fit_transform(data.reshape(-1, 1)).flatten()
        
        dataset = LightCurveDataset(normalized_data, sequence_length)
        train_size = int(0.8 * len(dataset))
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, len(dataset) - train_size]
        )
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        model = LSTMPredictor().to(self.device)
        criterion = nn.MSELoss()
        optimizer = Adam(model.parameters())
        
        best_val_loss = float('inf')
        for epoch in range(num_epochs):
            model.train()
            train_loss = 0
            for sequences, targets in train_loader:
                sequences = sequences.unsqueeze(-1).to(self.device)
                targets = targets.unsqueeze(-1).to(self.device)
                
                optimizer.zero_grad()
                output = model(sequences)
                loss = criterion(output, targets)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            # Validation
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for sequences, targets in val_loader:
                    sequences = sequences.unsqueeze(-1).to(self.device)
                    targets = targets.unsqueeze(-1).to(self.device)
                    output = model(sequences)
                    val_loss += criterion(output, targets).item()
            
            val_loss /= len(val_loader)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
            
            if (epoch + 1) % 10 == 0:
                logger.info(f"Epoch {epoch+1}/{num_epochs}, Validation Loss: {val_loss:.6f}")
        
        return model, best_val_loss
        
    def detect_anomalies(self, data: Union[np.ndarray, MaskedNDArray], contamination: float = 0.1) -> np.ndarray:
        """
        Detect anomalies in the time series using Isolation Forest.
        """
        # Convert masked array to regular numpy array
        if isinstance(data, MaskedNDArray):
            data = data.filled(np.nan)
        
        # Remove NaN values for anomaly detection
        valid_mask = ~np.isnan(data)
        valid_data = data[valid_mask]
        
        if len(valid_data) == 0:
            logger.warning("No valid data points for anomaly detection")
            return np.zeros(len(data), dtype=bool)
        
        detector = IsolationForest(contamination=contamination, random_state=42)
        anomalies = detector.fit_predict(valid_data.reshape(-1, 1))
        
        # Create full anomaly array including NaN positions
        full_anomalies = np.zeros(len(data), dtype=bool)
        full_anomalies[valid_mask] = (anomalies == -1)
        
        return full_anomalies
        
    def analyze_star(self, star_data: StarData) -> Dict[str, Any]:
        """
        Perform comprehensive analysis on stellar data.
        
        Args:
            star_data: StarData object containing light curve data
            
        Returns:
            Dictionary containing analysis results
        """
        logger.info(f"Analyzing star {star_data.star_id}")
        
        # Calculate time differences for sample rate
        time_diffs = np.diff(star_data.time)
        sample_rate = 1 / np.median(time_diffs)
        
        # Calculate entropy
        entropy = self.calculate_entropy(star_data.flux)
        
        # Perform frequency analysis
        freq_analysis = self.perform_frequency_analysis(star_data.flux, sample_rate)
        
        # Train LSTM model
        model, prediction_error = self.train_lstm_model(star_data.flux)
        
        # Detect anomalies
        anomalies = self.detect_anomalies(star_data.flux)
        
        results = {
            "star_id": star_data.star_id,
            "entropy": entropy,
            "frequency_analysis": freq_analysis,
            "prediction_error": prediction_error,
            "anomaly_percentage": np.mean(anomalies) * 100
        }
        
        # Save results
        results_file = self.output_dir / f"{star_data.star_id}_analysis.npz"
        np.savez(results_file, **results)
        
        return results
        
    def plot_results(self, star_data: StarData, results: Dict[str, Any]) -> None:
        """Create visualization of analysis results."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Convert masked arrays to regular arrays for plotting
        time = star_data.time.filled(np.nan) if isinstance(star_data.time, MaskedNDArray) else star_data.time
        flux = star_data.flux.filled(np.nan) if isinstance(star_data.flux, MaskedNDArray) else star_data.flux
        
        # Plot light curve
        ax1.plot(time, flux)
        ax1.set_title("Light Curve")
        ax1.set_xlabel("Time")
        ax1.set_ylabel("Flux")
        
        # Plot frequency spectrum
        freq_analysis = results["frequency_analysis"]
        ax2.plot(freq_analysis["frequencies"], freq_analysis["amplitudes"])
        ax2.set_title("Frequency Spectrum")
        ax2.set_xlabel("Frequency")
        ax2.set_ylabel("Amplitude")
        
        # Plot dominant frequencies
        ax3.scatter(freq_analysis["dominant_frequencies"],
                   freq_analysis["amplitudes"][np.array([np.where(freq_analysis["frequencies"] == f)[0][0]
                                                       for f in freq_analysis["dominant_frequencies"]])])
        ax3.set_title("Dominant Frequencies")
        ax3.set_xlabel("Frequency")
        ax3.set_ylabel("Amplitude")
        
        # Add text with summary statistics
        stats_text = (f"Entropy: {results['entropy']:.2f}\n"
                     f"Prediction Error: {results['prediction_error']:.2e}\n"
                     f"Anomaly %: {results['anomaly_percentage']:.2f}")
        ax4.text(0.1, 0.5, stats_text, transform=ax4.transAxes)
        ax4.axis('off')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f"{star_data.star_id}_plots.png")
        plt.close()

def main():
    """Main function to run the analysis."""
    # Example usage
    analyzer = StellarConsciousnessAnalyzer()
    
    # List of interesting variable stars
    target_stars = [
        "TIC 260128333",  # A known variable star
        "TIC 277539431",  # Another interesting target
    ]
    
    for star_id in target_stars:
        try:
            # Fetch and analyze star data
            star_data = analyzer.fetch_star_data(star_id)
            results = analyzer.analyze_star(star_data)
            
            # Plot results
            analyzer.plot_results(star_data, results)
            
            logger.info(f"Analysis completed for {star_id}")
            logger.info(f"Results: {results}")
            
        except Exception as e:
            logger.error(f"Error analyzing star {star_id}: {str(e)}")
            continue

if __name__ == "__main__":
    main()
