"""
HiC-SuperNet Loss Functions & Metrics
=======================================
Multi-component loss and evaluation metrics for Hi-C contact map enhancement.
"""

import tensorflow as tf
import numpy as np


# ============================================================================
# LOSS FUNCTION
# ============================================================================

def improved_loss(y_true, y_pred):
    
    y_true = tf.reshape(y_true, [-1, tf.shape(y_true)[1], tf.shape(y_true)[2], 1])
    y_pred = tf.reshape(y_pred, [-1, tf.shape(y_pred)[1], tf.shape(y_pred)[2], 1])

    # MSE
    mse_loss = tf.reduce_mean(tf.square(y_true - y_pred))

    # MAE
    mae_loss = tf.reduce_mean(tf.abs(y_true - y_pred))

    # Pearson Correlation Loss
    y_true_flat = tf.reshape(y_true, [tf.shape(y_true)[0], -1])
    y_pred_flat = tf.reshape(y_pred, [tf.shape(y_pred)[0], -1])

    y_true_centered = y_true_flat - tf.reduce_mean(y_true_flat, axis=1, keepdims=True)
    y_pred_centered = y_pred_flat - tf.reduce_mean(y_pred_flat, axis=1, keepdims=True)

    numerator   = tf.reduce_sum(y_true_centered * y_pred_centered, axis=1)
    denominator = tf.sqrt(
        tf.reduce_sum(tf.square(y_true_centered), axis=1) *
        tf.reduce_sum(tf.square(y_pred_centered), axis=1)
    )
    pearson_loss = 1.0 - tf.reduce_mean(numerator / (denominator + 1e-8))

    # SSIM Loss
    ssim_loss = 1.0 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1.0))

    return 0.4 * mse_loss + 0.2 * mae_loss + 0.3 * pearson_loss + 0.1 * ssim_loss


# ============================================================================
# INDIVIDUAL METRICS
# ============================================================================

def calculate_ssim(y_true, y_pred):
    
    return tf.image.ssim(y_true, y_pred, max_val=1.0)


def calculate_psnr(y_true, y_pred, max_val=1.0):
    
    return tf.image.psnr(y_true, y_pred, max_val=max_val)


def calculate_mse(y_true, y_pred):
    
    return tf.reduce_mean(tf.square(y_true - y_pred))


def calculate_mae(y_true, y_pred):
    
    return tf.reduce_mean(tf.abs(y_true - y_pred))


def calculate_pcc(y_true, y_pred):
   
    y_true_flat = tf.reshape(y_true, [-1])
    y_pred_flat = tf.reshape(y_pred, [-1])

    y_true_c = y_true_flat - tf.reduce_mean(y_true_flat)
    y_pred_c = y_pred_flat - tf.reduce_mean(y_pred_flat)

    numerator   = tf.reduce_sum(y_true_c * y_pred_c)
    denominator = tf.sqrt(
        tf.reduce_sum(tf.square(y_true_c)) *
        tf.reduce_sum(tf.square(y_pred_c))
    )
    return numerator / (denominator + 1e-8)


# ============================================================================
# AGGREGATE METRICS
# ============================================================================

def calculate_all_metrics(y_true, y_pred):
    
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    return {
        'SSIM': float(tf.reduce_mean(calculate_ssim(y_true, y_pred)).numpy()),
        'PSNR': float(tf.reduce_mean(calculate_psnr(y_true, y_pred)).numpy()),
        'MSE':  float(calculate_mse(y_true, y_pred).numpy()),
        'MAE':  float(calculate_mae(y_true, y_pred).numpy()),
        'PCC':  float(calculate_pcc(y_true, y_pred).numpy()),
    }


def print_metrics(metrics: dict, prefix: str = ''):
    """Pretty-print a metrics dictionary."""
    label = f"[{prefix}] " if prefix else ''
    print(f"  {label}SSIM : {metrics['SSIM']:.4f}")
    print(f"  {label}PSNR : {metrics['PSNR']:.2f} dB")
    print(f"  {label}MSE  : {metrics['MSE']:.6f}")
    print(f"  {label}MAE  : {metrics['MAE']:.6f}")
    print(f"  {label}PCC  : {metrics['PCC']:.4f}")
