# semantic_communication/channel.py

import torch
import torch.nn as nn
import math

class IdealChannel(nn.Module):
    """Represents a perfect, noiseless channel."""
    def __init__(self):
        super().__init__()
        print("Initialized IdealChannel")

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Input passes through unchanged."""
        return z

class AWGNChannel(nn.Module):
    """Additive White Gaussian Noise Channel."""
    def __init__(self, snr_db: float, input_power: float = 1.0):
        """
        Args:
            snr_db (float): Signal-to-Noise Ratio in decibels.
            input_power (float): Assumed average power of the input signal 'z'.
                                 If z is normalized or its power varies, this needs adjustment
                                 or calculation within the forward pass. Default assumes unit power.
        """
        super().__init__()
        if snr_db is None:
             print("Warning: snr_db is None for AWGNChannel. Assuming noiseless.")
             self.snr_db = float('inf') # Treat None as infinite SNR (noiseless)
             self.noise_stddev = 0.0
        else:
            self.snr_db = snr_db
            # Calculate noise standard deviation from SNR
            # SNR (linear) = Signal Power / Noise Power
            # Noise Power = Signal Power / SNR (linear)
            # Noise Power = Noise Variance = stddev^2 (for complex Gaussian, split power between real/imag)
            # For real-valued noise added to real 'z':
            snr_linear = 10 ** (self.snr_db / 10.0)
            noise_variance = input_power / snr_linear
            self.noise_stddev = math.sqrt(noise_variance)
            # Clamp stddev to avoid issues with extremely high SNR -> near zero stddev
            # self.noise_stddev = max(self.noise_stddev, 1e-9)

        print(f"Initialized AWGNChannel (SNR={self.snr_db} dB, Assumed Input Power={input_power})")
        if self.noise_stddev > 0:
            print(f"  Calculated Noise StdDev: {self.noise_stddev:.4f}")
        else:
             print(f"  No noise will be added (StdDev = 0.0)")


    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Adds AWGN to the input tensor.

        Args:
            z (torch.Tensor): Input latent tensor (batch_size, latent_dim).

        Returns:
            torch.Tensor: Output tensor 'z_prime' with added noise (batch_size, latent_dim).
        """
        if self.noise_stddev == 0.0 or not self.training:
            # No noise added if stddev is zero or if not in training mode
            # (often desirable to evaluate performance without noise sometimes)
            # However, for evaluating robustness, you might want noise during eval too.
            # Consider adding a flag `add_noise_in_eval` if needed.
            # For now, let's assume noise is primarily for training regularization/realism
             if not self.training and self.noise_stddev > 0.0 :
                 print("AWGNChannel: In eval mode, noise not added by default.")
             return z


        # Generate noise with the same shape and device as the input
        noise = torch.randn_like(z) * self.noise_stddev
        z_prime = z + noise
        return z_prime

class RayleighChannel(nn.Module):
    """Rayleigh Fading Channel with AWGN."""
    def __init__(self, snr_db: float, add_awgn: bool = True):
        """
        Args:
            snr_db (float): Average Signal-to-Noise Ratio in decibels, considering fading.
            add_awgn (bool): Whether to add AWGN after fading. If False, only fading is applied.
        """
        super().__init__()
        self.add_awgn = add_awgn
        if snr_db is None and add_awgn:
             print("Warning: snr_db is None for RayleighChannel with AWGN. Assuming noiseless AWGN.")
             self.snr_db = float('inf')
             self.noise_stddev = 0.0
        elif add_awgn:
            self.snr_db = snr_db
            # SNR = E[|h|^2] * Signal Power / Noise Power
            # For Rayleigh, E[|h|^2] = 1 (if properly normalized: h ~ CN(0, 1))
            # Noise Power = Signal Power / SNR (linear)
            # Assuming average input_power = 1
            snr_linear = 10 ** (self.snr_db / 10.0)
            noise_variance = 1.0 / snr_linear # Assumes signal power = 1
            self.noise_stddev = math.sqrt(noise_variance)
            # self.noise_stddev = max(self.noise_stddev, 1e-9)
        else:
             self.snr_db = None # No SNR relevant if no AWGN
             self.noise_stddev = 0.0

        print(f"Initialized RayleighChannel (Average SNR={self.snr_db} dB, Add AWGN={self.add_awgn})")
        if self.add_awgn and self.noise_stddev > 0:
             print(f"  Calculated Noise StdDev: {self.noise_stddev:.4f}")
        elif self.add_awgn:
             print(f"  No AWGN will be added (StdDev = 0.0)")


    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Applies Rayleigh fading and optionally AWGN.

        Args:
            z (torch.Tensor): Input latent tensor (batch_size, latent_dim).

        Returns:
            torch.Tensor: Output tensor 'z_prime' after fading and noise.
        """
        batch_size, latent_dim = z.shape
        device = z.device

        # Generate Rayleigh fading coefficients 'h'
        # h = (1/sqrt(2)) * (randn() + j*randn())
        # |h| follows Rayleigh distribution, E[|h|^2] = 1
        # We apply the same fading coeff across the latent dim for simplicity here,
        # could also apply independent fading per dimension if needed.
        h_real = torch.randn((batch_size, 1), device=device) / math.sqrt(2)
        h_imag = torch.randn((batch_size, 1), device=device) / math.sqrt(2)
        # For real-valued z, we typically only use the magnitude |h|
        h_mag = torch.sqrt(h_real**2 + h_imag**2) # Shape: (batch_size, 1)

        # Apply fading
        z_faded = h_mag * z # Element-wise multiplication, broadcast h_mag

        if not self.add_awgn:
            return z_faded

        # Add AWGN (if enabled and in training mode)
        if self.noise_stddev == 0.0 or not self.training:
            if not self.training and self.noise_stddev > 0.0:
                 print("RayleighChannel: In eval mode, AWGN not added by default.")
            return z_faded

        noise = torch.randn_like(z_faded) * self.noise_stddev
        z_prime = z_faded + noise
        return z_prime


# Helper function to select channel based on config
def get_channel(config: dict) -> nn.Module:
    """Instantiates the channel based on configuration."""
    channel_type = config.get('type', 'ideal').lower()
    snr_db = config.get('snr_db', None) # Allow None for ideal or no-AWGN cases

    if channel_type == 'ideal':
        return IdealChannel()
    elif channel_type == 'awgn':
        # input_power = config.get('channel_input_power', 1.0) # Could make this configurable
        return AWGNChannel(snr_db=snr_db)
    elif channel_type == 'rayleigh':
        add_awgn = config.get('rayleigh_add_awgn', True)
        return RayleighChannel(snr_db=snr_db, add_awgn=add_awgn)
    else:
        raise ValueError(f"Unknown channel type: {channel_type}")


# Example usage
if __name__ == '__main__':
    dummy_z = torch.randn(10, 64) # Example latent vectors

    print("\n--- Ideal Channel ---")
    ideal_channel = get_channel({'type': 'ideal'})
    z_prime_ideal = ideal_channel(dummy_z)
    print(f"Input z mean power: {torch.mean(dummy_z**2):.4f}")
    print(f"Output z' mean power: {torch.mean(z_prime_ideal**2):.4f}")
    print(f"Difference norm: {torch.norm(dummy_z - z_prime_ideal):.4f}") # Should be 0

    print("\n--- AWGN Channel (SNR=10dB) ---")
    awgn_channel = get_channel({'type': 'awgn', 'snr_db': 10})
    awgn_channel.train() # Need to be in training mode to add noise
    z_prime_awgn = awgn_channel(dummy_z)
    print(f"Input z mean power: {torch.mean(dummy_z**2):.4f}")
    print(f"Output z' mean power: {torch.mean(z_prime_awgn**2):.4f}")
    print(f"Noise power (z'-z): {torch.mean((z_prime_awgn - dummy_z)**2):.4f}") # Should be approx 1/10 = 0.1
    awgn_channel.eval() # Check eval mode
    z_prime_awgn_eval = awgn_channel(dummy_z)
    print(f"Difference norm (eval): {torch.norm(dummy_z - z_prime_awgn_eval):.4f}") # Should be 0

    print("\n--- Rayleigh Channel (SNR=5dB, with AWGN) ---")
    rayleigh_channel = get_channel({'type': 'rayleigh', 'snr_db': 5})
    rayleigh_channel.train()
    z_prime_rayleigh = rayleigh_channel(dummy_z)
    print(f"Input z mean power: {torch.mean(dummy_z**2):.4f}")
    print(f"Output z' mean power: {torch.mean(z_prime_rayleigh**2):.4f}") # Power should change due to fading+noise
    # Noise calculation is more complex here due to fading gain

    print("\n--- Rayleigh Channel (Fading only) ---")
    rayleigh_fade_only = get_channel({'type': 'rayleigh', 'snr_db': None, 'rayleigh_add_awgn': False})
    rayleigh_fade_only.train()
    z_prime_fade = rayleigh_fade_only(dummy_z)
    print(f"Input z mean power: {torch.mean(dummy_z**2):.4f}")
    print(f"Output z' mean power: {torch.mean(z_prime_fade**2):.4f}") # Power should change due to fading