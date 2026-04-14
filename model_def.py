import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Tuple, List

# ============================================================================
# 1. DATA PROCESSING PIPELINE (Used for inference data preparation)
# ============================================================================

class PongFrameProcessor:
    """Preprocesses Pong game frames: resize, grayscale, normalize."""

    def __init__(self, target_size: Tuple[int, int] = (64, 64), normalize_range: str = '0_1'):
        """
        Args:
            target_size: (H, W) tuple for resizing.
            normalize_range: '0_1' or '-1_1'.
        """
        self.target_size = target_size
        self.normalize_range = normalize_range

    def process_frame(self, frame_path: str) -> np.ndarray:
        """
        Process a single frame: resize, grayscale, normalize.

        Args:
            frame_path: Path to image file

        Returns:
            Processed frame as numpy array (H, W)
        """
        # Load image
        img = Image.open(frame_path)

        # Resize using bilinear interpolation
        img = img.resize(self.target_size, Image.Resampling.BILINEAR)

        # Convert to grayscale
        img = img.convert('L')

        # Convert to numpy array
        frame = np.array(img, dtype=np.float32)

        # Normalize
        if self.normalize_range == '0_1':
            frame = frame / 255.0
        elif self.normalize_range == '-1_1':
            frame = (frame / 127.5) - 1.0

        return frame

# NOTE: PongSequenceDataset is typically only needed for preparing test/validation sequences
# from a full directory, but is included for completeness if sequence loading is required.
class PongSequenceDataset(torch.utils.data.Dataset):
    """Dataset for Pong frame sequences (used mostly during training/evaluation)."""

    def __init__(self, data_dir: str, sequence_length: int = 10,
                   split: str = 'all', train_ratio: float = 0.7,
                   val_ratio: float = 0.15, processor: PongFrameProcessor = None):

        self.data_dir = Path(data_dir)
        self.sequence_length = sequence_length
        self.processor = processor or PongFrameProcessor()

        # Get all frame files sorted by name
        self.frame_files = sorted(list(self.data_dir.glob('*.png')) +
                                  list(self.data_dir.glob('*.jpg')))

        if len(self.frame_files) == 0:
            print(f"Warning: No image files found in {data_dir}")

        total_sequences = max(0, len(self.frame_files) - sequence_length)

        # Determine split indices (simplified logic from original for deployment context)
        train_end = int(total_sequences * train_ratio)
        val_end = int(total_sequences * (train_ratio + val_ratio))

        if split == 'train':
            self.sequence_indices = list(range(0, train_end))
        elif split == 'val':
            self.sequence_indices = list(range(train_end, val_end))
        elif split == 'test':
            self.sequence_indices = list(range(val_end, total_sequences))
        elif split == 'all':
             self.sequence_indices = list(range(0, total_sequences))
        else:
            raise ValueError("split must be 'train', 'val', 'test', or 'all'")


    def __len__(self) -> int:
        return len(self.sequence_indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            input_sequence: Tensor of shape (sequence_length, 1, H, W)
            target_frame: Tensor of shape (1, H, W)
        """
        start_idx = self.sequence_indices[idx]

        # Load input sequence
        input_frames = []
        for i in range(self.sequence_length):
            frame = self.processor.process_frame(
                str(self.frame_files[start_idx + i])
            )
            input_frames.append(frame)

        # Load target frame (the frame immediately following the sequence)
        target_frame = self.processor.process_frame(
            str(self.frame_files[start_idx + self.sequence_length])
        )

        # Convert to tensors
        # Sequence: (T, C, H, W). Target: (C, H, W)
        input_sequence = torch.tensor(input_frames, dtype=torch.float32).unsqueeze(1)
        target_frame = torch.tensor(target_frame, dtype=torch.float32).unsqueeze(0)

        return input_sequence, target_frame


# ============================================================================
# 2. MODEL 1 ARCHITECTURE: ConvLSTM
# ============================================================================

class ConvLSTMCell(nn.Module):
    """Convolutional LSTM Cell implementation."""

    def __init__(self, input_channels: int, hidden_channels: int,
                 kernel_size: int = 3):
        super().__init__()

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2

        # Gates: input, forget, output, cell (4 * hidden_channels)
        self.conv = nn.Conv2d(
            in_channels=input_channels + hidden_channels,
            out_channels=4 * hidden_channels,
            kernel_size=kernel_size,
            padding=self.padding
        )

    def forward(self, x: torch.Tensor, state: Tuple[torch.Tensor, torch.Tensor]):
        """
        Args:
            x: Input tensor (B, C, H, W)
            state: Tuple of (h, c) each (B, hidden_channels, H, W)

        Returns:
            h_next, c_next: Next hidden and cell states
        """
        h, c = state

        # Concatenate input and hidden state
        combined = torch.cat([x, h], dim=1)

        # Compute gates
        gates = self.conv(combined)

        # Split into 4 gates
        i, f, o, g = torch.split(gates, self.hidden_channels, dim=1)

        # Apply activations
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)

        # Update cell and hidden state
        c_next = f * c + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size: int, image_size: Tuple[int, int],
                    device: torch.device):
        """Initialize hidden and cell states (h and c) to zeros."""
        h, w = image_size
        h_state = torch.zeros(batch_size, self.hidden_channels, h, w, device=device)
        c_state = torch.zeros(batch_size, self.hidden_channels, h, w, device=device)
        return h_state, c_state


class ConvLSTM(nn.Module):
    """ConvLSTM Network for Frame Prediction."""

    def __init__(self, input_channels: int = 1, hidden_channels: List[int] = [64, 64, 64],
                 kernel_size: int = 3, output_channels: int = 1):
        super().__init__()

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.num_layers = len(hidden_channels)

        # Create ConvLSTM layers
        self.cells = nn.ModuleList()

        for i in range(self.num_layers):
            in_ch = input_channels if i == 0 else hidden_channels[i-1]
            self.cells.append(
                ConvLSTMCell(in_ch, hidden_channels[i], kernel_size)
            )

        # Output convolution to map the last hidden state back to the image channel size
        self.output_conv = nn.Conv2d(
            hidden_channels[-1], output_channels,
            kernel_size=1, padding=0
        )

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: Input sequence (B, T, C, H, W)

        Returns:
            prediction: Predicted frame (B, C, H, W)
        """
        batch_size, seq_len, _, h, w = x.size()
        device = x.device

        # Initialize hidden states for all layers
        states = []
        for i in range(self.num_layers):
            states.append(
                self.cells[i].init_hidden(batch_size, (h, w), device)
            )

        # Process sequence
        for t in range(seq_len):
            x_t = x[:, t] # Input frame for this timestep

            for layer_idx in range(self.num_layers):
                # Pass input through the layer
                h_next, c_next = self.cells[layer_idx](x_t, states[layer_idx])

                # Update state
                states[layer_idx] = (h_next, c_next)

                # Output of current layer becomes input of the next layer
                x_t = h_next

        # Generate prediction from final hidden state of the last layer
        output = self.output_conv(states[-1][0])

        return output


# ============================================================================
# 3. MODEL 2 ARCHITECTURE: BetaVAE and LatentRNN
# ============================================================================

class BetaVAE(nn.Module):
    """β-Variational Autoencoder for frame compression/decompression."""

    def __init__(self, input_channels: int = 1, latent_dim: int = 64,
                 beta: float = 4.0):
        super().__init__()

        self.latent_dim = latent_dim
        self.beta = beta

        # Encoder (64x64 -> 4x4)
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=4, stride=2, padding=1),  # 64->32
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # 32->16
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # 16->8
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),  # 8->4
            nn.ReLU(),
        )

        # Latent space: 256 * 4 * 4 = 4096 features flattened
        self.fc_mu = nn.Linear(256 * 4 * 4, latent_dim)
        self.fc_logvar = nn.Linear(256 * 4 * 4, latent_dim)

        # Decoder input
        self.decoder_input = nn.Linear(latent_dim, 256 * 4 * 4)

        # Decoder (4x4 -> 64x64)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # 4->8
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # 8->16
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # 16->32
            nn.ReLU(),
            nn.ConvTranspose2d(32, input_channels, kernel_size=4, stride=2, padding=1),  # 32->64
            nn.Sigmoid()  # Output in [0, 1]
        )

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode input to latent distribution parameters (mu, logvar)."""
        h = self.encoder(x)
        h = h.view(h.size(0), -1)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick for sampling z."""
        std = torch.exp(0.5 * logvar)
        # Use torch.randn_like for safety if different devices are involved
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent vector z to image."""
        h = self.decoder_input(z)
        h = h.view(h.size(0), 256, 4, 4)
        return self.decoder(h)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Input: frame tensor (B, C, H, W)
        Returns: reconstruction, mu, logvar
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstruction = self.decode(z)
        return reconstruction, mu, logvar

    def loss_function(self, recon_x: torch.Tensor, x: torch.Tensor,
                      mu: torch.Tensor, logvar: torch.Tensor) -> dict:
        """Compute β-VAE loss (Recon + beta * KL). Used mainly during training."""
        # Reconstruction loss (MSE)
        recon_loss = F.mse_loss(recon_x, x, reduction='sum')

        # KL divergence: KL(N(mu, sigma^2) || N(0, 1))
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        # Total loss
        total_loss = recon_loss + self.beta * kl_loss

        return {
            'total_loss': total_loss,
            'recon_loss': recon_loss,
            'kl_loss': kl_loss
        }


class LatentRNN(nn.Module):
    """Standard LSTM/RNN for sequence prediction in the latent space."""

    def __init__(self, latent_dim: int = 64, hidden_dim: int = 256,
                 num_layers: int = 2, rnn_type: str = 'lstm'):
        super().__init__()

        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.rnn_type = rnn_type.lower()

        if self.rnn_type == 'lstm':
            self.rnn = nn.LSTM(
                input_size=latent_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True
            )
        elif self.rnn_type == 'gru':
            self.rnn = nn.GRU(
                input_size=latent_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True
            )
        else:
            raise ValueError("rnn_type must be 'lstm' or 'gru'")

        # Fully connected layer to map RNN hidden state back to latent space
        self.fc = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: Latent sequences (B, T, latent_dim)

        Returns:
            prediction: Predicted next latent vector (B, latent_dim)
        """
        # lstm_out: (batch_size, seq_len, hidden_dim)
        lstm_out, _ = self.rnn(x)

        # Take the output from the last time step
        last_out = lstm_out[:, -1, :]

        # Predict the next latent vector
        prediction = self.fc(last_out)
        return prediction


# ============================================================================
# 4. INFERENCE FUNCTIONS
# ============================================================================

def generate_frames_convlstm(model: ConvLSTM, initial_sequence: torch.Tensor,
                             num_frames: int = 20, device: str = 'cuda') -> torch.Tensor:
    """
    Generate frames autoregressively using a trained ConvLSTM model.

    Args:
        model: Trained ConvLSTM instance.
        initial_sequence: Starting sequence tensor (1, seq_len, C, H, W).
        num_frames: Number of future frames to generate.
        device: Device to run on.

    Returns:
        A tensor containing all generated frames (num_frames, C, H, W).
    """

    model.eval()
    model = model.to(device)

    # Initial sequence: (1, seq_len, C, H, W)
    current_sequence = initial_sequence.to(device)
    generated_frames = []

    with torch.no_grad():
        for _ in range(num_frames):
            # Predict next frame
            # The model internally handles hidden state initialization based on input
            next_frame = model(current_sequence)
            generated_frames.append(next_frame.cpu())

            # Update sequence for the next prediction: remove oldest frame, add newest prediction
            # next_frame is (1, C, H, W). Unsqueeze(1) makes it (1, 1, C, H, W)
            next_frame_expanded = next_frame.unsqueeze(1)
            current_sequence = torch.cat([
                current_sequence[:, 1:],          # Remove first frame
                next_frame_expanded               # Add predicted frame
            ], dim=1)

    # Concatenate all generated frames into a single tensor
    return torch.cat(generated_frames, dim=0)


def generate_frames_vae_rnn(vae_model: BetaVAE, rnn_model: LatentRNN,
                            initial_sequence: torch.Tensor,
                            num_frames: int = 20, device: str = 'cuda') -> torch.Tensor:
    """
    Generate frames autoregressively using a VAE+RNN model structure.

    Args:
        vae_model: Trained BetaVAE instance.
        rnn_model: Trained LatentRNN instance.
        initial_sequence: Starting sequence tensor (1, seq_len, C, H, W).
        num_frames: Number of future frames to generate.
        device: Device to run on.

    Returns:
        A tensor containing all generated frames (num_frames, C, H, W).
    """
    vae_model.eval()
    rnn_model.eval()
    vae_model = vae_model.to(device)
    rnn_model = rnn_model.to(device)

    # initial_sequence is (1, T, C, H, W)
    T = initial_sequence.size(1)
    current_sequence = initial_sequence.to(device)
    generated_frames = []

    with torch.no_grad():
        # 1. Encode the initial sequence into latent space
        # Flatten the sequence: (1 * T, C, H, W)
        flat_frames = current_sequence.view(-1, current_sequence.size(2), current_sequence.size(3), current_sequence.size(4))
        
        # Get mean latent vectors for the sequence
        # latent_mus is (T, latent_dim)
        latent_mus, _ = vae_model.encode(flat_frames)
        
        # Reshape to sequence format: (1, T, latent_dim)
        current_latent_seq = latent_mus.unsqueeze(0)

        for _ in range(num_frames):
            # 2. Predict the next latent vector using the RNN
            # predicted_latent_z is (1, latent_dim)
            predicted_latent_mu = rnn_model(current_latent_seq)

            # NOTE: We can skip the reparameterization trick here for simplicity
            # in inference, using mu as the predicted latent state.
            
            # 3. Decode the predicted latent vector back to a frame
            # next_frame is (1, C, H, W)
            next_frame = vae_model.decode(predicted_latent_mu)
            generated_frames.append(next_frame.cpu())

            # 4. Update the latent sequence
            
            # Option A (Simpler but less accurate for VAE):
            # Use the predicted_latent_mu as the next input to the sequence.
            predicted_latent_mu_expanded = predicted_latent_mu.unsqueeze(1) # (1, 1, latent_dim)
            
            current_latent_seq = torch.cat([
                current_latent_seq[:, 1:],          # Remove first latent vector
                predicted_latent_mu_expanded        # Add predicted latent vector
            ], dim=1)

            # Option B (More robust):
            # Encode the newly generated 'next_frame' back to latent space (mu)
            # This ensures the latent sequence remains aligned with the visual output,
            # though it adds encoding overhead. Sticking to Option A for simpler
            # latent-space autoregression as often done in VAE-RNN models.
            
    # Concatenate all generated frames into a single tensor
    return torch.cat(generated_frames, dim=0)
