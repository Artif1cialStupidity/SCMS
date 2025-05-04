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
            snr_linear = 10 ** (self.snr_db / 10.0)
            noise_variance = input_power / snr_linear
            self.noise_stddev = math.sqrt(noise_variance)

        print(f"Initialized AWGNChannel (SNR={self.snr_db} dB, Assumed Input Power={input_power})")
        if self.noise_stddev > 0:
            print(f"  Calculated Noise StdDev: {self.noise_stddev:.4f}")
        else:
             print(f"  No noise will be added (StdDev = 0.0)")


    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Adds AWGN to the input tensor. Now always adds noise if stddev > 0.

        Args:
            z (torch.Tensor): Input latent tensor (batch_size, latent_dim).

        Returns:
            torch.Tensor: Output tensor 'z_prime' with added noise (batch_size, latent_dim).
        """
        if self.noise_stddev == 0.0: # Only skip if configured SNR is infinite/stddev is 0
             # print("AWGNChannel: No noise configured (StdDev = 0.0)") # Optional: keep for info
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
            # Assuming average input_power = 1
            snr_linear = 10 ** (self.snr_db / 10.0)
            noise_variance = 1.0 / snr_linear # Assumes signal power = 1
            self.noise_stddev = math.sqrt(noise_variance)
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
        Applies Rayleigh fading and optionally AWGN. Now always applies effects if configured.

        Args:
            z (torch.Tensor): Input latent tensor (batch_size, latent_dim).

        Returns:
            torch.Tensor: Output tensor 'z_prime' after fading and noise.
        """
        batch_size, latent_dim = z.shape
        device = z.device

        # Generate Rayleigh fading coefficients 'h'
        h_real = torch.randn((batch_size, 1), device=device) / math.sqrt(2)
        h_imag = torch.randn((batch_size, 1), device=device) / math.sqrt(2)
        h_mag = torch.sqrt(h_real**2 + h_imag**2) # Shape: (batch_size, 1)

        # Apply fading
        z_faded = h_mag * z # Element-wise multiplication, broadcast h_mag

        if not self.add_awgn:
            return z_faded

        # Add AWGN (if enabled and stddev > 0)
        if self.noise_stddev == 0.0:
            # print("RayleighChannel: No AWGN configured (StdDev = 0.0)") # Optional
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
    # Test in both modes, noise should be added if stddev > 0
    awgn_channel.train()
    z_prime_awgn_train = awgn_channel(dummy_z)
    awgn_channel.eval()
    z_prime_awgn_eval = awgn_channel(dummy_z)
    print(f"Input z mean power: {torch.mean(dummy_z**2):.4f}")
    print(f"Output z' (train) mean power: {torch.mean(z_prime_awgn_train**2):.4f}")
    print(f"Output z' (eval) mean power: {torch.mean(z_prime_awgn_eval**2):.4f}") # Should be similar to train
    print(f"Noise power (train): {torch.mean((z_prime_awgn_train - dummy_z)**2):.4f}")
    print(f"Noise power (eval): {torch.mean((z_prime_awgn_eval - dummy_z)**2):.4f}") # Should be similar


    print("\n--- Rayleigh Channel (SNR=5dB, with AWGN) ---")
    rayleigh_channel = get_channel({'type': 'rayleigh', 'snr_db': 5})
    rayleigh_channel.train()
    z_prime_rayleigh_train = rayleigh_channel(dummy_z)
    rayleigh_channel.eval()
    z_prime_rayleigh_eval = rayleigh_channel(dummy_z)
    print(f"Input z mean power: {torch.mean(dummy_z**2):.4f}")
    print(f"Output z' (train) mean power: {torch.mean(z_prime_rayleigh_train**2):.4f}")
    print(f"Output z' (eval) mean power: {torch.mean(z_prime_rayleigh_eval**2):.4f}")