# In analyze_spectrum.py

import torch
import numpy as np
import matplotlib.pyplot as plt
from torch_geometric.loader import DataLoader
import argparse

# --- Adjust these imports to match your project structure ---
# These are the standard import paths for GraphGPS.
from graphgps.utility.utils import load_cfg, set_cfg
from graphgps.model_builder import create_model
from graphgps.loader.dataset.factory import create_dataset

def marcenko_pastur_pdf(x, ratio, sigma=1.0):
    """Computes the Marčenko-Pastur probability density function."""
    c = ratio
    # Handle the case where c > 1, where there's a point mass at 0
    # The continuous part is the same, though.
    a = sigma**2 * (1 - np.sqrt(c))**2
    b = sigma**2 * (1 + np.sqrt(c))**2
    
    pdf = np.zeros_like(x)
    mask = (x >= a) & (x <= b)
    if np.any(mask):
        pdf[mask] = np.sqrt((b - x[mask]) * (x[mask] - a)) / (2 * np.pi * sigma**2 * c * x[mask])
    return pdf

def analyze(cfg_file, ckpt_path, dataset_split='test'):
    """
    Main analysis function.
    """
    # --- Load Config, Dataset, and Trained Model ---
    cfg = load_cfg(cfg_file)
    set_cfg(cfg)

    datasets = create_dataset()
    if dataset_split == 'train':
        data_loader = DataLoader(datasets[0], batch_size=cfg.train.batch_size, shuffle=False)
    elif dataset_split == 'val':
        data_loader = DataLoader(datasets[1], batch_size=cfg.train.batch_size, shuffle=False)
    else: # test
        data_loader = DataLoader(datasets[2], batch_size=cfg.train.batch_size, shuffle=False)

    model = create_model(datasets)
    model.load_state_dict(torch.load(ckpt_path)['model_state'])
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    print(f"Model Architecture:\n{model}")

    # --- Hook Setup ---
    activations = {}
    def get_activation(name):
        def hook(model, input, output):
            activations[name] = output.detach().cpu()
        return hook

    # *** CRITICAL STEP: Identify the layer to analyze ***
    # You MUST inspect your model architecture (printed above) to find the correct layer path.
    # The path below is a common example for GraphGPS, but it might be different for you.
    # Good candidates: `model.layers[k].gnn` or `model.layers[k].self_attn`
    try:
        # Example path for a GPSLayer's GNN block
        hook_layer_path_str = 'model.layers[4].gnn' 
        hook_layer = model.layers[4].gnn 
        print(f"Hooking layer: {hook_layer_path_str}")
    except (AttributeError, IndexError) as e:
        print(f"ERROR: Could not find the specified layer path. Please inspect the model architecture and edit the 'hook_layer' variable in the script.")
        print(e)
        return
        
    hook_handle = hook_layer.register_forward_hook(get_activation('target_layer'))

    # --- Feature Extraction ---
    print(f"Extracting features from the '{dataset_split}' dataset split...")
    feature_matrix_list = []
    with torch.no_grad():
        for batch in data_loader:
            batch.to(device)
            model(batch)
            feature_matrix_list.append(activations['target_layer'])
    
    feature_matrix_full = torch.cat(feature_matrix_list, dim=0)
    hook_handle.remove()

    N, d = feature_matrix_full.shape
    print(f"Extraction complete. Full feature matrix Φ has shape: {N} nodes x {d} hidden dimensions.")

    # --- Gram Matrix and Eigenvalue Computation ---
    print("Constructing Gram Matrix and computing eigenvalues...")
    # We use the (d x d) covariance matrix: (1/N) * Φ^T * Φ
    gram_matrix = (1 / N) * feature_matrix_full.T @ feature_matrix_full
    eigenvalues = torch.linalg.eigvalsh(gram_matrix).numpy()
    print(f"Computed {len(eigenvalues)} eigenvalues.")

    # --- Visualization ---
    plt.figure(figsize=(12, 7))
    plt.hist(eigenvalues, bins=50, density=True, label='Empirical Eigenvalues (from GNN)', color='skyblue', edgecolor='black')

    # Theoretical Marčenko-Pastur comparison
    c = N / d
    # The theory assumes entries with variance sigma^2. We estimate this from the features.
    sigma_sq = feature_matrix_full.var().item()
    
    x_min = 0 if c > 1 else (1 - np.sqrt(c))**2 * sigma_sq
    x_max = (1 + np.sqrt(c))**2 * sigma_sq
    
    x = np.linspace(x_min, x_max, 1000)
    mp_pdf = marcenko_pastur_pdf(x, c, sigma=np.sqrt(sigma_sq))
    plt.plot(x, mp_pdf, 'r-', linewidth=2, label=f'Marčenko-Pastur Law (c={c:.2f}, σ²={sigma_sq:.2f})')

    plt.title(f'Eigenvalue Spectrum of Gram Matrix\nLayer: {hook_layer_path_str}', fontsize=16)
    plt.xlabel('Eigenvalue', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Analyze the eigenvalue spectrum of a trained GraphGPS model.')
    parser.add_argument('--cfg', dest='cfg_file', type=str, required=True, help='Path to the configuration file (e.g., configs/GCN/zinc-GCN.yaml)')
    parser.add_argument('--ckpt', dest='ckpt_path', type=str, required=True, help='Path to the model checkpoint file (e.g., results/GCN/zinc-GCN/ckpt_best.pt)')
    parser.add_argument('--split', dest='dataset_split', type=str, default='test', choices=['train', 'val', 'test'], help='Which dataset split to analyze')
    args = parser.parse_args()
    
    analyze(args.cfg_file, args.ckpt_path, args.dataset_split)
