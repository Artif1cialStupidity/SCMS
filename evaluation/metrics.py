# evaluation/metrics.py

import torch
import torch.nn.functional as F
import math
# For SSIM, you might need to install scikit-image: pip install scikit-image
# from skimage.metrics import structural_similarity as ssim_calc # Optional
# For LPIPS, you might need to install lpips: pip install lpips
# import lpips # Optional

def calculate_accuracy(output_logits: torch.Tensor, target: torch.Tensor) -> tuple[float, int]:
    """
    Calculates classification accuracy.

    Args:
        output_logits (torch.Tensor): Model output logits (batch_size, num_classes).
        target (torch.Tensor): Ground truth class indices (batch_size).

    Returns:
        tuple[float, int]: Average accuracy (0.0 to 1.0) and number of correct predictions.
    """
    with torch.no_grad():
        probabilities = F.softmax(output_logits, dim=1)
        predicted_classes = torch.argmax(probabilities, dim=1)
        correct = (predicted_classes == target).sum().item()
        total = target.size(0)
        accuracy = correct / total
    return accuracy, correct

def calculate_psnr(img1: torch.Tensor, img2: torch.Tensor, data_range: float = 1.0) -> float:
    """
    Calculates Peak Signal-to-Noise Ratio between two batches of images.
    Assumes images are in the range [0, 1] or [-1, 1] after potential denormalization.
    The 'data_range' should match the maximum possible pixel value minus minimum.
    If images are normalized to [-1, 1] (like with Tanh output), data_range=2.0.
    If images are normalized to [0, 1] (like with Sigmoid output), data_range=1.0.
    Our CIFAR normalization puts data roughly in [-1.x, 1.x], but Tanh outputs [-1, 1].
    Let's assume the model outputs in [-1, 1] for consistency with Tanh. data_range=2.0
    If using MSE loss directly on normalized data, PSNR might be less intuitive.
    Often PSNR is calculated on images scaled back to [0, 255] or [0, 1].

    Args:
        img1 (torch.Tensor): First batch of images (B, C, H, W). Values in [-1, 1].
        img2 (torch.Tensor): Second batch of images (B, C, H, W). Values in [-1, 1].
        data_range (float): The range of the pixel values (max - min). Default 2.0 for [-1, 1].

    Returns:
        float: Average PSNR value in dB across the batch. Returns +inf if images are identical.
    """
    with torch.no_grad():
        # Ensure images are float
        img1 = img1.float()
        img2 = img2.float()

        # Calculate MSE
        mse = F.mse_loss(img1, img2, reduction='mean')

        if mse == 0:
            return float('inf') # Or a very large number like 100.0

        # Calculate PSNR
        psnr = 10.0 * math.log10((data_range ** 2) / mse.item())
        # Alternative using torch.log10
        # psnr_torch = 10.0 * torch.log10((data_range ** 2) / mse)
        # return psnr_torch.item()

    return psnr

# --- Optional Metrics (Require additional libraries) ---

def calculate_ssim(img1: torch.Tensor, img2: torch.Tensor, data_range: float = 1.0) -> float:
    """
    Calculates Structural Similarity Index (SSIM) between two batches of images.
    Requires scikit-image. Input images typically expected in [0, 1] or [0, 255].
    Need to handle data range and channel dimension correctly.

    Args:
        img1 (torch.Tensor): First batch (B, C, H, W). Assumed range [0, 1] for calculation.
        img2 (torch.Tensor): Second batch (B, C, H, W). Assumed range [0, 1] for calculation.
        data_range (float): Range of the input data (e.g., 1.0 for [0, 1]).

    Returns:
        float: Average SSIM value across the batch.
    """
    # TODO: Implement SSIM - Needs careful handling of:
    # 1. Data range conversion (e.g., from [-1, 1] to [0, 1] if needed)
    # 2. Permuting dimensions (skimage expects channel last or multichannel=True)
    # 3. Looping through batch or vectorizing the calculation
    # Example stub:
    with torch.no_grad():
        img1_np = img1.permute(0, 2, 3, 1).cpu().numpy() # B, H, W, C
        img2_np = img2.permute(0, 2, 3, 1).cpu().numpy()
        # Assume input was [-1, 1], convert to [0, 1]
        img1_np = (img1_np + 1.0) / 2.0
        img2_np = (img2_np + 1.0) / 2.0
        data_range = 1.0 # Range is now 1.0

        ssim_total = 0.0
        for i in range(img1_np.shape[0]):
             # For multichannel images (like RGB)
             ssim_val = ssim_calc(img1_np[i], img2_np[i], data_range=data_range, channel_axis=-1, win_size=7) # win_size might need adjustment for 32x32
             ssim_total += ssim_val
        avg_ssim = ssim_total / img1_np.shape[0]
    return avg_ssim

#Placeholder for LPIPS - Requires lpips library and potentially a pre-trained model
def get_lpips_model(net_type='vgg', device=torch.device('cpu')):
    """Helper to load LPIPS model once."""
    # Avoid loading the model repeatedly during evaluation
    # Could be stored globally or passed around.
    # Requires: pip install lpips
    try:
        import lpips
        model = lpips.LPIPS(net=net_type).to(device)
        model.eval() # Set to eval mode
        print(f"Loaded LPIPS model (net={net_type})")
        return model
    except ImportError:
        print("LPIPS library not found. Install with 'pip install lpips'")
        return None

lpips_model = None # Global placeholder

def calculate_lpips(img1: torch.Tensor, img2: torch.Tensor, lpips_net) -> float:
    """
    Calculates Learned Perceptual Image Patch Similarity (LPIPS).
    Requires lpips library and a loaded network model.
    Input images typically expected in range [-1, 1].

    Args:
        img1 (torch.Tensor): First batch (B, C, H, W), range [-1, 1].
        img2 (torch.Tensor): Second batch (B, C, H, W), range [-1, 1].
        lpips_net: The pre-loaded LPIPS network instance.

    Returns:
        float: Average LPIPS distance across the batch (lower is better).
    """
    if lpips_net is None:
        print("LPIPS model not available for calculation.")
        return float('nan')

    with torch.no_grad():
        # Ensure inputs are on the same device as the lpips model
        device = next(lpips_net.parameters()).device
        img1 = img1.to(device)
        img2 = img2.to(device)

        distances = lpips_net(img1, img2) # Calculate distances
        avg_distance = distances.mean().item() # Average over batch

    return avg_distance


# --- Metrics for Attack Evaluation ---

def calculate_model_agreement(output_victim: torch.Tensor, output_surrogate: torch.Tensor, task: str) -> float:
    """
    Calculates the agreement rate between victim and surrogate model outputs.

    Args:
        output_victim (torch.Tensor): Victim model output (logits or reconstructed image).
        output_surrogate (torch.Tensor): Surrogate model output (logits or reconstructed image).
        task (str): 'classification' or 'reconstruction'.

    Returns:
        float: Agreement rate (0.0 to 1.0). For reconstruction, this might be less meaningful
               unless thresholded difference is used. Primarily for classification.
    """
    with torch.no_grad():
        if task == 'classification':
            pred_victim = torch.argmax(output_victim, dim=1)
            pred_surrogate = torch.argmax(output_surrogate, dim=1)
            agreement = (pred_victim == pred_surrogate).float().mean().item()
        elif task == 'reconstruction':
            # Agreement for reconstruction is ill-defined.
            # Could calculate MSE or L1 distance instead.
            # Returning 0 for now, indicating it's not standard for reconstruction.
            print("Warning: Model agreement is not typically used for reconstruction tasks.")
            agreement = 0.0
            # Alternatively, calculate something like proportion of pixels within a threshold?
            # threshold = 0.1 * data_range # Example threshold
            # pixel_agreement = (torch.abs(output_victim - output_surrogate) < threshold).float().mean().item()
            # agreement = pixel_agreement
        else:
             raise ValueError(f"Unknown task for agreement: {task}")
    return agreement

# --- Example Usage ---
if __name__ == '__main__':
    # Test Accuracy
    print("--- Testing Accuracy ---")
    dummy_logits = torch.randn(10, 5) # Batch 10, 5 classes
    dummy_targets = torch.randint(0, 5, (10,))
    acc, correct = calculate_accuracy(dummy_logits, dummy_targets)
    print(f"Targets: {dummy_targets}")
    print(f"Predicted: {torch.argmax(dummy_logits, dim=1)}")
    print(f"Accuracy: {acc*100:.2f}%, Correct: {correct}")

    # Test PSNR
    print("\n--- Testing PSNR ---")
    img_a = torch.rand(4, 3, 32, 32) * 2 - 1 # Range [-1, 1]
    img_b = img_a + torch.randn_like(img_a) * 0.1 # Add some noise
    img_b = torch.clamp(img_b, -1, 1)
    img_c = img_a.clone() # Identical image

    psnr_ab = calculate_psnr(img_a, img_b, data_range=2.0)
    psnr_ac = calculate_psnr(img_a, img_c, data_range=2.0)
    print(f"PSNR (Noisy vs Original): {psnr_ab:.2f} dB")
    print(f"PSNR (Identical vs Original): {psnr_ac}") # Should be inf

    # Test Model Agreement (Classification)
    print("\n--- Testing Model Agreement ---")
    vic_logits = torch.tensor([[0.1, 0.9], [0.8, 0.2], [0.4, 0.6]])
    sur_logits_match = torch.tensor([[0.2, 0.8], [0.6, 0.4], [0.3, 0.7]]) # Same predictions
    sur_logits_mismatch = torch.tensor([[0.7, 0.3], [0.6, 0.4], [0.1, 0.9]]) # Different predictions

    agree_match = calculate_model_agreement(vic_logits, sur_logits_match, 'classification')
    agree_mismatch = calculate_model_agreement(vic_logits, sur_logits_mismatch, 'classification')
    print(f"Agreement (Matching): {agree_match*100:.2f}%") # Should be 100%
    print(f"Agreement (Mismatching): {agree_mismatch*100:.2f}%") # Should be ~33% (1/3 match)

    # Test Model Agreement (Reconstruction - returns 0 or placeholder)
    print("\n--- Testing Model Agreement (Reconstruction) ---")
    recon_agree = calculate_model_agreement(img_a, img_b, 'reconstruction')
    print(f"Agreement (Reconstruction): {recon_agree}")

    # Optional: Test SSIM/LPIPS if libraries are installed and code uncommented
    # print("\n--- Testing Optional Metrics ---")
    # if ssim_calc:
    #     # Ensure images are in [0, 1] for SSIM testing
    #     img_a_01 = (img_a + 1.0) / 2.0
    #     img_b_01 = (img_b + 1.0) / 2.0
    #     ssim_ab = calculate_ssim(img_a_01, img_b_01, data_range=1.0)
    #     print(f"SSIM (Noisy vs Original): {ssim_ab:.4f}")
    # if lpips:
    #     lpips_net_instance = get_lpips_model()
    #     if lpips_net_instance:
    #          lpips_ab = calculate_lpips(img_a, img_b, lpips_net_instance)
    #          print(f"LPIPS (Noisy vs Original): {lpips_ab:.4f} (Lower is better)")