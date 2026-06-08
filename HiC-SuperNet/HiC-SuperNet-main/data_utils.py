
import os
import numpy as np
import tensorflow as tf


# ============================================================================
# LOW-LEVEL HELPERS
# ============================================================================

def _channels_last(arr: np.ndarray) -> np.ndarray:
    
    if arr.ndim == 4 and arr.shape[1] == 1:
        arr = np.transpose(arr, (0, 2, 3, 1))
    elif arr.ndim == 3:
        arr = np.expand_dims(arr, axis=-1)
    elif arr.ndim == 4 and arr.shape[-1] == 1:
        pass  # already correct
    else:
        raise ValueError(
            f"Unexpected array shape {arr.shape}. "
            "Expected (N,1,H,W), (N,H,W), or (N,H,W,1)."
        )
    return arr


def _normalize(arr: np.ndarray) -> np.ndarray:

    lo, hi = arr.min(), arr.max()
    return (arr - lo) / (hi - lo + 1e-8)


# ============================================================================
# DATA LOADER
# ============================================================================

def load_npz(path: str):
    
    if not os.path.exists(path):
        raise FileNotFoundError(f"Data file not found: {path}")

    npz = np.load(path)
    required = {'data', 'target'}
    missing  = required - set(npz.files)
    if missing:
        raise KeyError(f"Keys {missing} not found in {path}. Found: {npz.files}")

    X = npz['data'].astype(np.float32)
    y = npz['target'].astype(np.float32)

    X = _channels_last(X)
    y = _channels_last(y)

    return X, y


def load_hic_data(train_file: str,
                  valid_file: str,
                  test_file:  str,
                  normalize:  bool = True,
                  verbose:    bool = True):
   
    splits = {
        'train': train_file,
        'valid': valid_file,
        'test':  test_file,
    }

    if verbose:
        print('=' * 60)
        print('Loading Hi-C Data')
        print('=' * 60)

    loaded = {}
    for name, path in splits.items():
        if verbose:
            print(f'  [{name}] {path}')
        X, y = load_npz(path)

        if normalize:
            X = _normalize(X)
            y = _normalize(y)

        loaded[name] = (X, y)

        if verbose:
            print(f'         X {X.shape}  y {y.shape}  '
                  f'range [{X.min():.3f}, {X.max():.3f}]')

    if verbose:
        print('=' * 60)

    return loaded['train'], loaded['valid'], loaded['test']


# ============================================================================
# DATASET FACTORY
# ============================================================================

def create_tf_dataset(X:          np.ndarray,
                      y:          np.ndarray,
                      batch_size: int  = 16,
                      shuffle:    bool = True,
                      augment:    bool = False) -> tf.data.Dataset:
   
    dataset = tf.data.Dataset.from_tensor_slices((X, y))

    if shuffle:
        dataset = dataset.shuffle(buffer_size=min(len(X), 2000), reshuffle_each_iteration=True)

    if augment:
        def _augment(x, target):
            # Random horizontal flip (contact maps are symmetric, so this is valid)
            combined = tf.concat([x, target], axis=-1)   # (H, W, 2)
            combined = tf.image.random_flip_left_right(combined)
            combined = tf.image.random_flip_up_down(combined)
            x_aug      = combined[..., :1]
            target_aug = combined[..., 1:]
            return x_aug, target_aug
        dataset = dataset.map(_augment, num_parallel_calls=tf.data.AUTOTUNE)

    dataset = dataset.batch(batch_size, drop_remainder=False)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset


# ============================================================================
# DATA INSPECTION UTILITIES
# ============================================================================

def summarize_dataset(X: np.ndarray, y: np.ndarray, name: str = ''):
    """Print descriptive statistics for a dataset split."""
    tag = f'[{name}] ' if name else ''
    print(f'{tag}X : shape={X.shape}  min={X.min():.4f}  '
          f'max={X.max():.4f}  mean={X.mean():.4f}  std={X.std():.4f}')
    print(f'{tag}y : shape={y.shape}  min={y.min():.4f}  '
          f'max={y.max():.4f}  mean={y.mean():.4f}  std={y.std():.4f}')


def get_patch_size(X: np.ndarray) -> int:
    """Return the spatial dimension (H) of the input patches."""
    return X.shape[1]


# ============================================================================
# PREPROCESSING SCRIPT  
# ============================================================================

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Verify and inspect HiC-SuperNet .npz data files.'
    )
    parser.add_argument('--train', required=True, help='Path to training .npz')
    parser.add_argument('--valid', required=True, help='Path to validation .npz')
    parser.add_argument('--test',  required=True, help='Path to test .npz')
    args = parser.parse_args()

    (train_X, train_y), (valid_X, valid_y), (test_X, test_y) = load_hic_data(
        args.train, args.valid, args.test
    )

    print('\nDetailed statistics:')
    summarize_dataset(train_X, train_y, 'train')
    summarize_dataset(valid_X, valid_y, 'valid')
    summarize_dataset(test_X,  test_y,  'test')

    print('\nPatch size:', get_patch_size(train_X))
    print('\nAll files loaded and validated successfully.')
