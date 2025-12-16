# ==================== SETUP & INSTALLATION ====================
print("=" * 60)
print("INSTALLING DEPENDENCIES...")
print("=" * 60)

import subprocess
import sys

def install_packages():
    packages = ['torch', 'librosa', 'numpy', 'scipy', 'soundfile']
    for package in packages:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', package])
    # Explicitly upgrade sympy to resolve potential compatibility issues with torch
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--upgrade', '-q', 'sympy'])

install_packages()

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import librosa
from torch.utils.data import Dataset, DataLoader
import os
from typing import List, Tuple
import math
import matplotlib.pyplot as plt
import soundfile as sf
from IPython.display import Audio, display
import warnings
import glob
warnings.filterwarnings('ignore')

print("✓ All packages installed successfully!")
print()

# ==================== MOUNT GOOGLE DRIVE ====================
print("=" * 60)
print("MOUNTING GOOGLE DRIVE...")
print("=" * 60)

from google.colab import drive
drive.mount('/content/drive')

print("✓ Google Drive mounted successfully!")
print()

# ==================== LOAD AUDIO FILES FROM DRIVE ====================
print("=" * 60)
print("LOADING AUDIO FILES FROM GOOGLE DRIVE...")
print("=" * 60)

# Path ke folder audio di Google Drive
AUDIO_FOLDER = '/content/drive/MyDrive/Agen cerdas/AudioK2'  # Sesuaikan dengan struktur folder Anda

# Cari semua file .wav
audio_files = sorted(glob.glob(os.path.join(AUDIO_FOLDER, '*.wav')))

if len(audio_files) == 0:
    print("❌ No audio files found!")
    print(f"Please check if the folder exists: {AUDIO_FOLDER}")
    print("\nTrying alternative paths...")

    # Coba beberapa path alternatif
    alternative_paths = [
        '/content/drive/MyDrive/audio',
        '/content/drive/My Drive/audio',
        '/content/drive/Shareddrives/*/audio'
    ]

    for path in alternative_paths:
        audio_files = sorted(glob.glob(os.path.join(path, '*.wav')))
        if len(audio_files) > 0:
            AUDIO_FOLDER = path
            print(f"✓ Found files in: {path}")
            break
else:
    print(f"✓ Found {len(audio_files)} audio files")
    print(f"✓ Location: {AUDIO_FOLDER}")
    print("\nFirst 5 files:")
    for i, f in enumerate(audio_files[:5]):
        print(f"  {i+1}. {os.path.basename(f)}")
    print()

# Display first audio file
if len(audio_files) > 0:
    print("Sample Audio File:")
    display(Audio(audio_files[0]))
    print()

# ==================== CREATE LABELS AUTOMATICALLY ====================
print("=" * 60)
print("CREATING AUTOMATIC LABELS...")
print("=" * 60)

def create_auto_labels(audio_path, sr=8000, frame_length=0.01):
    """
    Automatically create labels using energy-based voice activity detection
    and frequency separation for 2 speakers
    """
    # Load audio
    y, sr = librosa.load(audio_path, sr=sr)

    # Voice Activity Detection (VAD)
    frame_samples = int(frame_length * sr)
    energy = librosa.feature.rms(y=y, frame_length=frame_samples, hop_length=frame_samples)[0]

    # Normalize energy
    energy = (energy - np.min(energy)) / (np.max(energy) - np.min(energy) + 1e-8)

    # Threshold for activity
    threshold = 0.3
    activity = (energy > threshold).astype(float)

    # Separate speakers using frequency bands
    # Low frequency = Speaker 1, High frequency = Speaker 2

    # Low-pass filter for speaker 1
    from scipy import signal
    b_low, a_low = signal.butter(5, 800, 'low', fs=sr)
    y_low = signal.filtfilt(b_low, a_low, y)
    energy_low = librosa.feature.rms(y=y_low, frame_length=frame_samples, hop_length=frame_samples)[0]
    energy_low = (energy_low - np.min(energy_low)) / (np.max(energy_low) - np.min(energy_low) + 1e-8)

    # High-pass filter for speaker 2
    b_high, a_high = signal.butter(5, 800, 'high', fs=sr)
    y_high = signal.filtfilt(b_high, a_high, y)
    energy_high = librosa.feature.rms(y=y_high, frame_length=frame_samples, hop_length=frame_samples)[0]
    energy_high = (energy_high - np.min(energy_high)) / (np.max(energy_high) - np.min(energy_high) + 1e-8)

    # Assign speakers based on dominant frequency
    speaker1 = ((energy_low > energy_high) & (activity == 1)).astype(float)
    speaker2 = ((energy_high > energy_low) & (activity == 1)).astype(float)

    # Create label matrix
    labels = np.stack([speaker1, speaker2], axis=1)

    return labels

# Create labels for all audio files
print("Processing audio files to generate labels...")
label_data = []

for i, audio_file in enumerate(audio_files):
    try:
        labels = create_auto_labels(audio_file)
        label_data.append(labels)

        if (i + 1) % 5 == 0:
            print(f"  Processed {i+1}/{len(audio_files)} files...")
    except Exception as e:
        print(f"  ⚠ Error processing {os.path.basename(audio_file)}: {e}")
        # Create dummy labels if error
        dummy_labels = np.random.randint(0, 2, size=(500, 2))
        label_data.append(dummy_labels)

print(f"✓ Generated labels for {len(label_data)} audio files")
print()

# ==================== FEATURE EXTRACTION ====================
class AudioFeatureExtractor:
    """Extract mel-filterbank features from audio"""

    def __init__(self, sr=8000, n_mels=23, frame_length=0.025, frame_shift=0.01):
        self.sr = sr
        self.n_mels = n_mels
        self.n_fft = int(frame_length * sr)
        self.hop_length = int(frame_shift * sr)

    def extract_features(self, audio_path):
        """Extract log mel-filterbank energies"""
        y, sr = librosa.load(audio_path, sr=self.sr)

        mel_spec = librosa.feature.melspectrogram(
            y=y,
            sr=sr,
            n_mels=self.n_mels,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )

        log_mel = librosa.power_to_db(mel_spec, ref=np.max)
        features = log_mel.T

        return features

    def context_window(self, features, left=7, right=7):
        """Add context by concatenating neighboring frames"""
        T, D = features.shape
        padded = np.pad(features, ((left, right), (0, 0)), mode='edge')

        windowed = []
        for i in range(T):
            window = padded[i:i+left+right+1].flatten()
            windowed.append(window)

        return np.array(windowed)


# ==================== DATASET ====================
class SpeakerDiarizationDataset(Dataset):
    """Dataset for speaker diarization"""

    def __init__(self, audio_files, labels_list, feature_extractor=None):
        self.audio_files = audio_files
        self.labels_list = labels_list
        self.feature_extractor = feature_extractor or AudioFeatureExtractor()

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        # Extract features
        features = self.feature_extractor.extract_features(self.audio_files[idx])
        features = self.feature_extractor.context_window(features)
        features = features[::10]  # Temporal downsampling

        # Get labels
        labels = self.labels_list[idx]
        labels = labels[::10]  # Match downsampling

        # Ensure same length
        min_len = min(len(features), len(labels))
        features = features[:min_len]
        labels = labels[:min_len]

        return torch.FloatTensor(features), torch.FloatTensor(labels)


def collate_fn(batch):
    """
    Custom collate function to handle variable length sequences
    Pads sequences to the same length in a batch
    """
    features_list, labels_list = zip(*batch)

    # Find max length in this batch (limit to prevent OOM)
    max_len = max([f.shape[0] for f in features_list])
    MAX_SEQUENCE_LENGTH = 1000  # Limit max sequence length to prevent OOM
    max_len = min(max_len, MAX_SEQUENCE_LENGTH)

    feature_dim = features_list[0].shape[1]
    num_speakers = labels_list[0].shape[1]

    # Pad features and labels
    padded_features = []
    padded_labels = []

    for features, labels in zip(features_list, labels_list):
        seq_len = features.shape[0]

        # Truncate if too long
        if seq_len > max_len:
            features = features[:max_len]
            labels = labels[:max_len]
            seq_len = max_len

        # Pad features
        if seq_len < max_len:
            pad_len = max_len - seq_len
            features_pad = torch.zeros(pad_len, feature_dim)
            features = torch.cat([features, features_pad], dim=0)

        # Pad labels
        if seq_len < max_len:
            pad_len = max_len - seq_len
            labels_pad = torch.zeros(pad_len, num_speakers)
            labels = torch.cat([labels, labels_pad], dim=0)

        padded_features.append(features)
        padded_labels.append(labels)

    # Stack into batch
    features_batch = torch.stack(padded_features, dim=0)
    labels_batch = torch.stack(padded_labels, dim=0)

    return features_batch, labels_batch


# ==================== MODEL COMPONENTS ====================
class MultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention mechanism"""

    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, x):
        batch_size, seq_len, _ = x.size()

        Q = self.W_q(x).view(batch_size, seq_len, self.num_heads, self.d_k)
        K = self.W_k(x).view(batch_size, seq_len, self.num_heads, self.d_k)
        V = self.W_v(x).view(batch_size, seq_len, self.num_heads, self.d_k)

        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        attention_weights = F.softmax(scores, dim=-1)

        context = torch.matmul(attention_weights, V)
        context = context.transpose(1, 2).contiguous()
        context = context.view(batch_size, seq_len, self.d_model)

        output = self.W_o(context)

        return output, attention_weights


class TransformerEncoderBlock(nn.Module):
    """Transformer encoder block"""

    def __init__(self, d_model, num_heads, d_ff=2048, dropout=0.1):
        super().__init__()

        self.attention = MultiHeadSelfAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attn_out, attn_weights = self.attention(x)
        x = self.norm1(x + self.dropout(attn_out))

        ff_out = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_out))

        return x, attn_weights


class SAEEND(nn.Module):
    """Self-Attention End-to-End Neural Diarization"""

    def __init__(self, input_dim, d_model=256, num_blocks=4, num_heads=4,
                 num_speakers=2, dropout=0.1):
        super().__init__()

        self.num_speakers = num_speakers
        self.num_heads = num_heads
        self.num_blocks = num_blocks

        self.input_layer = nn.Linear(input_dim, d_model)
        self.input_norm = nn.LayerNorm(d_model)

        self.encoder_blocks = nn.ModuleList([
            TransformerEncoderBlock(d_model, num_heads, dropout=dropout)
            for _ in range(num_blocks)
        ])

        self.output_norm = nn.LayerNorm(d_model)
        self.output_layer = nn.Linear(d_model, num_speakers)

    def forward(self, x):
        x = self.input_layer(x)
        x = self.input_norm(x)

        all_attention_weights = []

        for block in self.encoder_blocks:
            x, attn_weights = block(x)
            all_attention_weights.append(attn_weights)

        x = self.output_norm(x)
        output = torch.sigmoid(self.output_layer(x))

        return output, all_attention_weights


# ==================== AUXILIARY LOSSES ====================
class AuxiliaryLosses:
    """Compute SVAD and OSD auxiliary losses"""

    @staticmethod
    def svad_loss(attention_weights, speaker_labels):
        """Speaker-wise VAD loss"""
        batch_size, num_heads, seq_len, _ = attention_weights.shape
        num_speakers = speaker_labels.shape[-1]

        total_loss = 0
        count = 0

        for s in range(num_speakers):
            y_s = speaker_labels[:, :, s]
            M_s = torch.bmm(y_s.unsqueeze(2), y_s.unsqueeze(1))

            traces = torch.diagonal(attention_weights, dim1=-2, dim2=-1).sum(dim=-1)
            head_idx = torch.argmax(traces, dim=1)

            for b in range(batch_size):
                A_h = attention_weights[b, head_idx[b]]
                loss = F.binary_cross_entropy(A_h, M_s[b])
                total_loss += loss
                count += 1

        return total_loss / count if count > 0 else torch.tensor(0.0)

    @staticmethod
    def osd_loss(attention_weights, speaker_labels, k=0.707):
        """Overlapped Speech Detection loss"""
        batch_size, seq_len, num_speakers = speaker_labels.shape

        speech_activity = speaker_labels.sum(dim=-1)

        osd_labels = torch.zeros_like(speech_activity)
        osd_labels[speech_activity == 1] = k
        osd_labels[speech_activity >= 2] = 1.0

        M_OSD = torch.bmm(osd_labels.unsqueeze(2), osd_labels.unsqueeze(1))

        traces = torch.diagonal(attention_weights, dim1=-2, dim2=-1).sum(dim=-1)
        head_idx = torch.argmax(traces, dim=1)

        total_loss = 0
        for b in range(batch_size):
            A_h = attention_weights[b, head_idx[b]]
            loss = F.mse_loss(A_h, M_OSD[b])
            total_loss += loss

        return total_loss / batch_size


# ==================== TRAINING ====================
def train_model(model, train_loader, num_epochs=30, lr=0.0001,
                alpha=1.0, beta=1.0, device='cpu'):
    """Train SA-EEND model with auxiliary losses"""

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    aux_losses = AuxiliaryLosses()

    history = {
        'total_loss': [],
        'diar_loss': [],
        'svad_loss': [],
        'osd_loss': []
    }

    print("=" * 60)
    print("TRAINING MODEL...")
    print("=" * 60)

    for epoch in range(num_epochs):
        model.train()
        epoch_total_loss = 0
        epoch_diar_loss = 0
        epoch_svad_loss = 0
        epoch_osd_loss = 0

        for batch_idx, (features, labels) in enumerate(train_loader):
            features = features.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            predictions, all_attention_weights = model(features)

            # Diarization loss
            diar_loss = F.binary_cross_entropy(predictions, labels)

            # Auxiliary losses (4th encoder block)
            attn_4th_block = all_attention_weights[-1]
            svad_loss = aux_losses.svad_loss(attn_4th_block, labels)
            osd_loss = aux_losses.osd_loss(attn_4th_block, labels)

            # Total loss
            loss = diar_loss + alpha * svad_loss + beta * osd_loss

            loss.backward()
            optimizer.step()

            epoch_total_loss += loss.item()
            epoch_diar_loss += diar_loss.item()
            epoch_svad_loss += svad_loss.item()
            epoch_osd_loss += osd_loss.item()

        # Average losses
        avg_total = epoch_total_loss / len(train_loader)
        avg_diar = epoch_diar_loss / len(train_loader)
        avg_svad = epoch_svad_loss / len(train_loader)
        avg_osd = epoch_osd_loss / len(train_loader)

        history['total_loss'].append(avg_total)
        history['diar_loss'].append(avg_diar)
        history['svad_loss'].append(avg_svad)
        history['osd_loss'].append(avg_osd)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d}/{num_epochs} | "
                  f"Loss: {avg_total:.4f} | "
                  f"Diar: {avg_diar:.4f} | "
                  f"SVAD: {avg_svad:.4f} | "
                  f"OSD: {avg_osd:.4f}")

    print("\n✓ Training completed!")
    return model, history


# ==================== VISUALIZATION ====================
def plot_training_history(history):
    """Plot training losses"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Training History', fontsize=16, fontweight='bold')

    axes[0, 0].plot(history['total_loss'], linewidth=2, color='purple')
    axes[0, 0].set_title('Total Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(history['diar_loss'], color='orange', linewidth=2)
    axes[0, 1].set_title('Diarization Loss')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(history['svad_loss'], color='green', linewidth=2)
    axes[1, 0].set_title('SVAD Loss (Speaker-wise VAD)')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(history['osd_loss'], color='red', linewidth=2)
    axes[1, 1].set_title('OSD Loss (Overlapped Speech Detection)')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Loss')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def visualize_predictions(audio_path, ground_truth, predictions, sr=8000):
    """Visualize diarization results"""

    # Load audio
    y, _ = librosa.load(audio_path, sr=sr)

    # Time axes
    time_gt = np.linspace(0, len(y)/sr, len(ground_truth))
    time_pred = np.linspace(0, len(y)/sr, len(predictions))

    fig, axes = plt.subplots(4, 1, figsize=(15, 10))
    fig.suptitle(f'Speaker Diarization: {os.path.basename(audio_path)}',
                 fontsize=16, fontweight='bold')

    # Waveform
    axes[0].plot(np.linspace(0, len(y)/sr, len(y)), y, linewidth=0.5, color='black')
    axes[0].set_title('Audio Waveform')
    axes[0].set_ylabel('Amplitude')
    axes[0].grid(True, alpha=0.3)

    # Ground truth - Speaker 1
    axes[1].fill_between(time_gt, 0, ground_truth[:, 0],
                         alpha=0.5, color='blue', label='Speaker 1')
    axes[1].set_title('Ground Truth - Speaker 1')
    axes[1].set_ylabel('Activity')
    axes[1].set_ylim([-0.1, 1.1])
    axes[1].grid(True, alpha=0.3)

    # Ground truth - Speaker 2
    axes[2].fill_between(time_gt, 0, ground_truth[:, 1],
                         alpha=0.5, color='red', label='Speaker 2')
    axes[2].set_title('Ground Truth - Speaker 2')
    axes[2].set_ylabel('Activity')
    axes[2].set_ylim([-0.1, 1.1])
    axes[2].grid(True, alpha=0.3)

    # Predictions
    axes[3].fill_between(time_pred, 0, predictions[:, 0],
                         alpha=0.5, color='blue', label='Speaker 1')
    axes[3].fill_between(time_pred, 0, predictions[:, 1],
                         alpha=0.5, color='red', label='Speaker 2')
    axes[3].set_title('Predicted Speaker Activities')
    axes[3].set_xlabel('Time (s)')
    axes[3].set_ylabel('Activity')
    axes[3].set_ylim([-0.1, 1.1])
    axes[3].legend(loc='upper right')
    axes[3].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


# ==================== INFERENCE ====================
def predict_diarization(model, audio_path, feature_extractor, device='cpu', threshold=0.5):
    """Predict speaker diarization"""

    model.eval()
    model = model.to(device)

    features = feature_extractor.extract_features(audio_path)
    features = feature_extractor.context_window(features)
    features = features[::10]

    features = torch.FloatTensor(features).unsqueeze(0).to(device)

    with torch.no_grad():
        predictions, _ = model(features)

    predictions = predictions.cpu().numpy()[0]
    predictions = (predictions > threshold).astype(float)

    return predictions


# ==================== RTTM GENERATION ====================
def predictions_to_rttm(predictions, audio_filename, frame_shift=0.1, min_duration=0.0):
    """
    Convert predictions to RTTM (Rich Transcription Time Marked) format

    RTTM Format:
    SPEAKER <file-id> <channel-id> <start-time> <duration> <NA> <NA> <speaker-id> <NA> <NA>

    Args:
        predictions: numpy array (T, num_speakers) with binary predictions
        audio_filename: name of audio file
        frame_shift: time between frames in seconds (default 0.1s = 100ms)
        min_duration: minimum segment duration to include (seconds)

    Returns:
        List of RTTM lines
    """
    rttm_lines = []
    num_speakers = predictions.shape[1]

    # Process each speaker
    for spk_idx in range(num_speakers):
        speaker_id = f"speaker_{spk_idx + 1}"
        activity = predictions[:, spk_idx]

        # Find segments where speaker is active
        in_segment = False
        segment_start = 0

        for frame_idx in range(len(activity)):
            if activity[frame_idx] == 1 and not in_segment:
                # Start of new segment
                in_segment = True
                segment_start = frame_idx
            elif activity[frame_idx] == 0 and in_segment:
                # End of segment
                in_segment = False
                segment_end = frame_idx

                # Calculate time
                start_time = segment_start * frame_shift
                duration = (segment_end - segment_start) * frame_shift

                # Only add if duration meets minimum threshold
                if duration >= min_duration:
                    # RTTM format: SPEAKER file 1 start duration <NA> <NA> spk <NA> <NA>
                    rttm_line = f"SPEAKER {audio_filename} 1 {start_time:.3f} {duration:.3f} <NA> <NA> {speaker_id} <NA> <NA>"
                    rttm_lines.append(rttm_line)

        # Handle case where segment extends to end
        if in_segment:
            segment_end = len(activity)
            start_time = segment_start * frame_shift
            duration = (segment_end - segment_start) * frame_shift

            if duration >= min_duration:
                rttm_line = f"SPEAKER {audio_filename} 1 {start_time:.3f} {duration:.3f} <NA> <NA> {speaker_id} <NA> <NA>"
                rttm_lines.append(rttm_line)

    return rttm_lines


def save_rttm(rttm_lines, output_path):
    """Save RTTM lines to file"""
    with open(output_path, 'w') as f:
        for line in rttm_lines:
            f.write(line + '\n')
    print(f"✓ RTTM saved to: {output_path}")


def save_all_rttm(model, audio_files, feature_extractor, output_dir, device='cpu'):
    """Generate and save RTTM files for all audio files"""

    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("GENERATING RTTM FILES...")
    print("=" * 60)

    for i, audio_path in enumerate(audio_files):
        # Get predictions
        predictions = predict_diarization(model, audio_path, feature_extractor, device=device)

        # Generate RTTM
        audio_basename = os.path.splitext(os.path.basename(audio_path))[0]
        rttm_lines = predictions_to_rttm(predictions, audio_basename)

        # Save RTTM
        rttm_filename = f"{audio_basename}.rttm"
        rttm_path = os.path.join(output_dir, rttm_filename)
        save_rttm(rttm_lines, rttm_path)

        if (i + 1) % 5 == 0:
            print(f"  Processed {i+1}/{len(audio_files)} files...")

    print(f"\n✓ All RTTM files saved to: {output_dir}")
    print(f"✓ Total files: {len(audio_files)}")

    return output_dir


# ==================== MAIN EXECUTION ====================
def main():
    if len(audio_files) == 0:
        print("❌ ERROR: No audio files found!")
        print("\nPlease make sure:")
        print("1. You've shared the Google Drive folder")
        print("2. The folder structure is correct")
        print("3. The audio files are in .wav format")
        return

    # Configuration
    NUM_SPEAKERS = 2
    BATCH_SIZE = 2  # Reduced from 4 to 2 to save memory
    NUM_EPOCHS = 30
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("=" * 60)
    print("CONFIGURATION")
    print("=" * 60)
    print(f"Device: {DEVICE}")
    print(f"Number of speakers: {NUM_SPEAKERS}")
    print(f"Batch size: {BATCH_SIZE} (reduced for memory)")
    print(f"Number of epochs: {NUM_EPOCHS}")
    print(f"Audio files: {len(audio_files)}")

    # Clear GPU cache before starting
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print()

    # Feature extractor
    feature_extractor = AudioFeatureExtractor()

    # Create dataset and dataloader
    dataset = SpeakerDiarizationDataset(audio_files, label_data, feature_extractor)
    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

    # Initialize model (smaller model to save memory)
    input_dim = 23 * 15  # n_mels * context_window
    model = SAEEND(
        input_dim=input_dim,
        d_model=128,  # Reduced from 256 to 128
        num_blocks=4,
        num_heads=4,
        num_speakers=NUM_SPEAKERS,
        dropout=0.1
    )

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")
    print()

    # Train model
    model, history = train_model(
        model,
        train_loader,
        num_epochs=NUM_EPOCHS,
        lr=0.0001,
        alpha=1.0,
        beta=1.0,
        device=DEVICE
    )

    # Plot training history
    print("\n" + "=" * 60)
    print("TRAINING RESULTS")
    print("=" * 60)
    plot_training_history(history)

    # Save model to Google Drive
    model_save_path = '/content/drive/MyDrive/Agen cerdas/Model/saeend_model.pth'
    torch.save(model.state_dict(), model_save_path)
    print(f"\n✓ Model saved to Google Drive: {model_save_path}")

    # Test on first 3 audio files
    print("\n" + "=" * 60)
    print("INFERENCE EXAMPLES")
    print("=" * 60)

    num_test = min(3, len(audio_files))
    for i in range(num_test):
        print(f"\n[{i+1}/{num_test}] Processing: {os.path.basename(audio_files[i])}")

        predictions = predict_diarization(
            model,
            audio_files[i],
            feature_extractor,
            device=DEVICE
        )

        # Calculate statistics
        total_frames = len(predictions)
        spk1_frames = predictions[:, 0].sum()
        spk2_frames = predictions[:, 1].sum()
        overlap_frames = (predictions[:, 0] * predictions[:, 1]).sum()

        print(f"  Total frames: {total_frames}")
        print(f"  Speaker 1 active: {int(spk1_frames)} frames ({spk1_frames/total_frames*100:.1f}%)")
        print(f"  Speaker 2 active: {int(spk2_frames)} frames ({spk2_frames/total_frames*100:.1f}%)")
        print(f"  Overlap: {int(overlap_frames)} frames ({overlap_frames/total_frames*100:.1f}%)")

        # Generate RTTM for this file
        audio_basename = os.path.splitext(os.path.basename(audio_files[i]))[0]
        rttm_lines = predictions_to_rttm(predictions, audio_basename)

        print(f"\n  📄 RTTM Output ({len(rttm_lines)} segments):")
        for line in rttm_lines[:5]:  # Show first 5 lines
            print(f"    {line}")
        if len(rttm_lines) > 5:
            print(f"    ... and {len(rttm_lines) - 5} more segments")

        # Visualize
        visualize_predictions(audio_files[i], label_data[i], predictions)

        # Play audio
        print(f"\n  🔊 Audio playback:")
        display(Audio(audio_files[i]))

    # Generate RTTM files for ALL audio files
    print("\n" + "=" * 60)
    print("GENERATING RTTM FILES FOR ALL AUDIO")
    print("=" * 60)

    rttm_output_dir = '/content/drive/MyDrive/Agen cerdas/RTTM Output'
    save_all_rttm(model, audio_files, feature_extractor, rttm_output_dir, device=DEVICE)

    print("\n" + "=" * 60)
    print("✓ ALL DONE!")
    print("=" * 60)
    print(f"\n📊 Summary:")
    print(f"  - Trained on {len(audio_files)} audio files")
    print(f"  - Final loss: {history['total_loss'][-1]:.4f}")
    print(f"  - Model saved to: {model_save_path}")
    print(f"  - RTTM files saved to: {rttm_output_dir}")
    print(f"  - Total RTTM files: {len(audio_files)}")
    print(f"\n📁 Output files:")
    print(f"  - Model: {model_save_path}")
    print(f"  - RTTM folder: {rttm_output_dir}")

# Run main
if __name__ == "__main__":
    main()
