
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf
from tqdm import tqdm

from model      import HiCSuperNetGenerator, build_model
from losses     import improved_loss, calculate_all_metrics, print_metrics
from data_utils import load_npz, _normalize, _channels_last, create_tf_dataset


# ============================================================================
# INFERENCE
# ============================================================================

def predict_batch(model, X: np.ndarray, batch_size: int = 16) -> np.ndarray:
    
    preds = []
    for start in tqdm(range(0, len(X), batch_size), desc='Predicting'):
        x_batch = X[start:start + batch_size]
        p       = model(x_batch, training=False).numpy()
        preds.append(p)
    return np.concatenate(preds, axis=0)


# ============================================================================
# EVALUATION
# ============================================================================

def evaluate(model,
             test_X:     np.ndarray,
             test_y:     np.ndarray,
             batch_size: int = 16,
             verbose:    bool = True):
    
    preds = predict_batch(model, test_X, batch_size)
    metrics = calculate_all_metrics(test_y, preds)

    if verbose:
        print(f"\n{'='*60}")
        print('Test Set Evaluation Results')
        print(f"{'='*60}")
        print_metrics(metrics, 'test')
        print(f"{'='*60}\n")

    return metrics, preds


# ============================================================================
# VISUALISATION HELPERS
# ============================================================================

def save_comparison_grid(test_X:    np.ndarray,
                         test_y:    np.ndarray,
                         preds:     np.ndarray,
                         out_path:  str,
                         num_samples: int = 8):
  
    n    = min(num_samples, len(test_X))
    idxs = np.random.choice(len(test_X), n, replace=False)

    fig, axes = plt.subplots(n, 3, figsize=(12, 4 * n))
    if n == 1:
        axes = axes[np.newaxis, :]

    for row, idx in enumerate(idxs):
        m = calculate_all_metrics(test_y[idx:idx+1], preds[idx:idx+1])
        panels = [
            (test_X[idx, :, :, 0], 'Input (Low-Res)'),
            (preds[idx,  :, :, 0], f'Predicted\nSSIM={m["SSIM"]:.3f}  PCC={m["PCC"]:.3f}'),
            (test_y[idx, :, :, 0], 'Ground Truth (Hi-Res)'),
        ]
        for col, (img, title) in enumerate(panels):
            axes[row, col].imshow(img, cmap='Reds', vmin=0, vmax=1)
            axes[row, col].set_title(title, fontsize=9, fontweight='bold')
            axes[row, col].axis('off')

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Comparison grid saved → {out_path}')


def save_difference_maps(test_y:   np.ndarray,
                         preds:    np.ndarray,
                         out_path: str,
                         num_samples: int = 5):
   
    n    = min(num_samples, len(test_y))
    idxs = np.random.choice(len(test_y), n, replace=False)

    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    if n == 1:
        axes = [axes]

    for col, idx in enumerate(idxs):
        diff = np.abs(preds[idx, :, :, 0] - test_y[idx, :, :, 0])
        im   = axes[col].imshow(diff, cmap='hot', vmin=0, vmax=0.2)
        axes[col].set_title(f'|Δ| sample {idx}', fontsize=9)
        axes[col].axis('off')
        plt.colorbar(im, ax=axes[col], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Difference maps saved → {out_path}')


def plot_metric_distribution(test_y:   np.ndarray,
                             preds:    np.ndarray,
                             out_path: str):
    
    ssim_vals, pcc_vals = [], []

    for i in range(len(test_y)):
        m = calculate_all_metrics(test_y[i:i+1], preds[i:i+1])
        ssim_vals.append(m['SSIM'])
        pcc_vals.append(m['PCC'])

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].hist(ssim_vals, bins=30, color='steelblue', edgecolor='white', alpha=0.85)
    axes[0].axvline(np.mean(ssim_vals), color='red', lw=2, ls='--',
                    label=f'mean={np.mean(ssim_vals):.4f}')
    axes[0].set(xlabel='SSIM', ylabel='Count', title='Per-sample SSIM Distribution')
    axes[0].legend()

    axes[1].hist(pcc_vals, bins=30, color='darkorange', edgecolor='white', alpha=0.85)
    axes[1].axvline(np.mean(pcc_vals), color='blue', lw=2, ls='--',
                    label=f'mean={np.mean(pcc_vals):.4f}')
    axes[1].set(xlabel='PCC', ylabel='Count', title='Per-sample PCC Distribution')
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Metric distributions saved → {out_path}')


# ============================================================================
# SAVE PREDICTIONS
# ============================================================================

def save_predictions(preds:    np.ndarray,
                     test_y:   np.ndarray,
                     test_X:   np.ndarray,
                     out_path: str):
    """Save predictions and inputs/targets to a .npz file."""
    np.savez_compressed(out_path,
                        predictions=preds,
                        targets=test_y,
                        inputs=test_X)
    print(f'Predictions saved → {out_path}')


# ============================================================================
# LOAD MODEL HELPER
# ============================================================================

def load_model(weights_path: str,
               input_shape:  tuple,
               base_filters: int = 64,
               num_blocks:   int = 8) -> HiCSuperNetGenerator:
  
    model = build_model(input_shape,
                        base_filters=base_filters,
                        num_blocks=num_blocks,
                        verbose=False)
    model.load_weights(weights_path)
    print(f'Loaded weights from → {weights_path}')
    return model


# ============================================================================
# CLI ENTRY POINT
# ============================================================================

def parse_args():
    p = argparse.ArgumentParser(description='Test / Inference for HiC-SuperNet')

    # Model
    p.add_argument('--weights',      required=True,
                   help='Path to trained weights (.weights.h5 or .h5)')
    p.add_argument('--base_filters', type=int, default=64)
    p.add_argument('--num_blocks',   type=int, default=8)

    # Data  (use --test for evaluation with ground truth)
    p.add_argument('--test',       default=None,
                   help='Path to test .npz (requires data + target keys)')
    p.add_argument('--input_npz',  default=None,
                   help='Path to input-only .npz (data key, no target) for pure inference')

    p.add_argument('--batch_size', type=int,   default=16)
    p.add_argument('--output',     default='results/',
                   help='Output directory for plots and saved predictions')
    p.add_argument('--num_vis',    type=int,   default=8,
                   help='Number of samples to visualise')
    p.add_argument('--seed',       type=int,   default=42)
    return p.parse_args()


def main():
    args = parse_args()
    tf.random.set_seed(args.seed)
    np.random.seed(args.seed)

    os.makedirs(args.output, exist_ok=True)

    # ── Load data ─────────────────────────────────────────────────────────────
    if args.test:
        test_X, test_y = load_npz(args.test)
        print(f'Test data loaded: {test_X.shape}')
        has_targets = True
    elif args.input_npz:
        npz = np.load(args.input_npz)
        test_X = npz['data'].astype(np.float32)
        test_X = _channels_last(test_X)
        test_X = _normalize(test_X)
        test_y = None
        has_targets = False
        print(f'Input data loaded (no targets): {test_X.shape}')
    else:
        raise ValueError('Provide --test or --input_npz')

    # ── Load model ────────────────────────────────────────────────────────────
    input_shape = test_X.shape[1:]
    model = load_model(args.weights, input_shape,
                       base_filters=args.base_filters,
                       num_blocks=args.num_blocks)

    # ── Evaluate / infer ──────────────────────────────────────────────────────
    if has_targets:
        metrics, preds = evaluate(model, test_X, test_y, args.batch_size)

        # Save numerical results
        results_path = os.path.join(args.output, 'test_metrics.txt')
        with open(results_path, 'w') as f:
            for k, v in metrics.items():
                f.write(f'{k}: {v}\n')
        print(f'Metrics saved → {results_path}')

        # Plots
        save_comparison_grid(test_X, test_y, preds,
                             os.path.join(args.output, 'comparison_grid.png'),
                             num_samples=args.num_vis)
        save_difference_maps(test_y, preds,
                             os.path.join(args.output, 'difference_maps.png'),
                             num_samples=min(5, args.num_vis))
        plot_metric_distribution(test_y, preds,
                                 os.path.join(args.output, 'metric_distributions.png'))
        save_predictions(preds, test_y, test_X,
                         os.path.join(args.output, 'predictions.npz'))
    else:
        preds = predict_batch(model, test_X, args.batch_size)
        out_path = os.path.join(args.output, 'enhanced_predictions.npz')
        np.savez_compressed(out_path, predictions=preds, inputs=test_X)
        print(f'Enhanced maps saved → {out_path}')

    print('\n✓ Done.')


if __name__ == '__main__':
    main()
