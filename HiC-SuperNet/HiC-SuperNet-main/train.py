

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')           # headless-safe backend
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tqdm import tqdm

from model      import HiCSuperNetGenerator, build_model
from losses     import improved_loss, calculate_all_metrics, print_metrics
from data_utils import load_hic_data, create_tf_dataset


# ============================================================================
# GPU SETUP
# ============================================================================

def configure_gpu():
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f'✓ GPU enabled: {len(gpus)} device(s)')
        except RuntimeError as e:
            print(f'⚠ GPU warning: {e}')
    else:
        print('⚠ No GPU found – running on CPU.')


# ============================================================================
# SINGLE STEP FUNCTIONS  (compiled with tf.function for speed)
# ============================================================================

@tf.function
def train_step(model, optimizer, loss_fn, x, y):
    with tf.GradientTape() as tape:
        y_pred = model(x, training=True)
        loss   = loss_fn(y, y_pred)
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss


@tf.function
def eval_step(model, loss_fn, x, y):
    y_pred = model(x, training=False)
    loss   = loss_fn(y, y_pred)
    return loss, y_pred


# ============================================================================
# TRAINING LOOP
# ============================================================================

def run_epoch(model, optimizer, loss_fn, dataset, training: bool, desc: str):
    """Run one full epoch. Returns (avg_loss, all_preds, all_targets)."""
    losses, preds, targets = [], [], []
    pbar = tqdm(dataset, desc=desc, leave=False)

    for x_batch, y_batch in pbar:
        if training:
            loss = train_step(model, optimizer, loss_fn, x_batch, y_batch)
            pbar.set_postfix({'loss': f'{loss.numpy():.6f}'})
            losses.append(loss.numpy())
        else:
            loss, y_pred = eval_step(model, loss_fn, x_batch, y_batch)
            losses.append(loss.numpy())
            preds.append(y_pred.numpy())
            targets.append(y_batch.numpy())

    avg_loss = float(np.mean(losses))
    if preds:
        all_preds   = np.concatenate(preds,   axis=0)
        all_targets = np.concatenate(targets, axis=0)
    else:
        all_preds = all_targets = None

    return avg_loss, all_preds, all_targets


def train(model,
          train_dataset,
          valid_dataset,
          test_dataset,
          epochs:         int   = 100,
          learning_rate:  float = 1e-3,
          checkpoint_dir: str   = 'checkpoints_hicsupernet',
          patience:       int   = 15):
   
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Cosine-decay learning rate schedule
    lr_schedule = keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=learning_rate,
        decay_steps=epochs,
        alpha=1e-6
    )
    optimizer = keras.optimizers.Adam(learning_rate=lr_schedule)

    history = {k: [] for k in ['train_loss', 'valid_loss',
                                'valid_ssim', 'valid_psnr', 'valid_pcc']}

    best_valid_loss  = float('inf')
    epochs_no_improve = 0

    print(f"\n{'='*60}")
    print('HiC-SuperNet Training')
    print(f"{'='*60}")
    print(f'  Max epochs    : {epochs}')
    print(f'  Learning rate : {learning_rate}  (cosine decay)')
    print(f'  Early stop    : patience={patience}')
    print(f'  Checkpoint dir: {checkpoint_dir}')
    print(f"{'='*60}\n")

    for epoch in range(1, epochs + 1):
        print(f"Epoch {epoch}/{epochs}")

        # ── Training ──────────────────────────────────────────────────────────
        train_loss, _, _ = run_epoch(
            model, optimizer, improved_loss, train_dataset,
            training=True, desc='  train'
        )

        # ── Validation ────────────────────────────────────────────────────────
        valid_loss, val_preds, val_targets = run_epoch(
            model, optimizer, improved_loss, valid_dataset,
            training=False, desc='  valid'
        )
        val_metrics = calculate_all_metrics(val_targets, val_preds)

        # ── Logging ───────────────────────────────────────────────────────────
        print(f'  train loss : {train_loss:.6f}')
        print(f'  valid loss : {valid_loss:.6f}')
        print_metrics(val_metrics, 'valid')

        history['train_loss'].append(train_loss)
        history['valid_loss'].append(valid_loss)
        history['valid_ssim'].append(val_metrics['SSIM'])
        history['valid_psnr'].append(val_metrics['PSNR'])
        history['valid_pcc'].append(val_metrics['PCC'])

        # ── Checkpoint ────────────────────────────────────────────────────────
        if valid_loss < best_valid_loss:
            best_valid_loss   = valid_loss
            epochs_no_improve = 0
            best_path = os.path.join(checkpoint_dir, 'best_model.weights.h5')
            model.save_weights(best_path)
            print(f'  ✓ Best model saved → {best_path}')
        else:
            epochs_no_improve += 1
            print(f'  No improvement ({epochs_no_improve}/{patience})')

        # Periodic snapshot
        if epoch % 10 == 0:
            snap_path = os.path.join(checkpoint_dir, f'epoch_{epoch:04d}.weights.h5')
            model.save_weights(snap_path)

        # ── Early stopping ────────────────────────────────────────────────────
        if epochs_no_improve >= patience:
            print(f'\nEarly stopping at epoch {epoch} (no improvement for {patience} epochs).')
            break

    # ── Final test evaluation ─────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print('FINAL TEST EVALUATION')
    print(f"{'='*60}")

    # Reload best weights
    model.load_weights(os.path.join(checkpoint_dir, 'best_model.weights.h5'))

    test_loss, test_preds, test_targets = run_epoch(
        model, optimizer, improved_loss, test_dataset,
        training=False, desc='  test'
    )
    test_metrics = calculate_all_metrics(test_targets, test_preds)

    print(f'  test loss : {test_loss:.6f}')
    print_metrics(test_metrics, 'test')

    # ── Persist artefacts ─────────────────────────────────────────────────────
    final_weights = os.path.join(checkpoint_dir, 'final_model.weights.h5')
    model.save_weights(final_weights)

    history_path = os.path.join(checkpoint_dir, 'training_history.npz')
    np.savez(history_path, **{k: np.array(v) for k, v in history.items()})

    print(f'\nSaved:')
    print(f'  {final_weights}')
    print(f'  {history_path}')

    return history, test_metrics


# ============================================================================
# VISUALISATION
# ============================================================================

def plot_training_history(history: dict, out_dir: str = '.'):
    """Save a 3-panel training history figure."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].plot(history['train_loss'], label='Train', lw=2)
    axes[0].plot(history['valid_loss'], label='Valid', lw=2)
    axes[0].set(xlabel='Epoch', ylabel='Loss', title='Training Loss')
    axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].plot(history['valid_ssim'], color='green', lw=2, label='SSIM')
    axes[1].set(xlabel='Epoch', ylabel='SSIM', title='Validation SSIM')
    axes[1].legend(); axes[1].grid(alpha=0.3)

    axes[2].plot(history['valid_psnr'], color='red', lw=2, label='PSNR')
    axes[2].set(xlabel='Epoch', ylabel='PSNR (dB)', title='Validation PSNR')
    axes[2].legend(); axes[2].grid(alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(out_dir, 'training_history.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Training history plot saved → {out_path}')


def visualize_predictions(model, test_X: np.ndarray, test_y: np.ndarray,
                          num_samples: int = 5, out_dir: str = '.'):
    """Save a grid of input / prediction / ground-truth comparisons."""
    indices = np.random.choice(len(test_X), min(num_samples, len(test_X)), replace=False)

    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4 * num_samples))
    if num_samples == 1:
        axes = axes[np.newaxis, :]

    from losses import calculate_all_metrics as _metrics

    for row, idx in enumerate(indices):
        x_s = test_X[idx:idx+1]
        y_s = test_y[idx:idx+1]
        p_s = model(x_s, training=False).numpy()
        m   = _metrics(y_s, p_s)

        for col, (data, title) in enumerate([
            (x_s[0, :, :, 0], 'Input (Low-Res)'),
            (p_s[0, :, :, 0], f'Predicted\nSSIM={m["SSIM"]:.3f}  PSNR={m["PSNR"]:.1f}'),
            (y_s[0, :, :, 0], 'Ground Truth (Hi-Res)'),
        ]):
            axes[row, col].imshow(data, cmap='Reds', vmin=0, vmax=1)
            axes[row, col].set_title(title, fontsize=10, fontweight='bold')
            axes[row, col].axis('off')

    plt.tight_layout()
    out_path = os.path.join(out_dir, 'predictions_sample.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Prediction visualisation saved → {out_path}')


# ============================================================================
# CLI ENTRY POINT
# ============================================================================

def parse_args():
    p = argparse.ArgumentParser(description='Train HiC-SuperNet')
    p.add_argument('--train',          default='hicarn_10kb40kb_c40_s40_b201_nonpool_train.npz')
    p.add_argument('--valid',          default='hicarn_10kb40kb_c40_s40_b201_nonpool_valid.npz')
    p.add_argument('--test',           default='hicarn_10kb40kb_c40_s40_b201_nonpool_GM12878_test.npz')
    p.add_argument('--epochs',         type=int,   default=100)
    p.add_argument('--batch_size',     type=int,   default=16)
    p.add_argument('--lr',             type=float, default=1e-3,
                   help='Initial learning rate (cosine-decayed)')
    p.add_argument('--base_filters',   type=int,   default=64)
    p.add_argument('--num_blocks',     type=int,   default=8)
    p.add_argument('--patience',       type=int,   default=15,
                   help='Early-stopping patience')
    p.add_argument('--augment',        action='store_true',
                   help='Apply random flips during training')
    p.add_argument('--checkpoint_dir', default='checkpoints_hicsupernet')
    p.add_argument('--seed',           type=int,   default=42)
    return p.parse_args()


def main():
    args = parse_args()

    tf.random.set_seed(args.seed)
    np.random.seed(args.seed)

    configure_gpu()

    # Load data
    (train_X, train_y), (valid_X, valid_y), (test_X, test_y) = load_hic_data(
        args.train, args.valid, args.test
    )

    # Build datasets
    train_ds = create_tf_dataset(train_X, train_y, args.batch_size,
                                  shuffle=True, augment=args.augment)
    valid_ds = create_tf_dataset(valid_X, valid_y, args.batch_size, shuffle=False)
    test_ds  = create_tf_dataset(test_X,  test_y,  args.batch_size, shuffle=False)

    # Build model
    input_shape = train_X.shape[1:]   # (H, W, 1)
    model = build_model(input_shape,
                        base_filters=args.base_filters,
                        num_blocks=args.num_blocks)

    # Train
    history, test_metrics = train(
        model,
        train_ds, valid_ds, test_ds,
        epochs=args.epochs,
        learning_rate=args.lr,
        checkpoint_dir=args.checkpoint_dir,
        patience=args.patience,
    )

    # Plots
    plot_training_history(history, out_dir=args.checkpoint_dir)
    visualize_predictions(model, test_X, test_y,
                          num_samples=5, out_dir=args.checkpoint_dir)

    print('\n✓ Training complete.')


if __name__ == '__main__':
    main()
