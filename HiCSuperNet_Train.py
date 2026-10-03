"""
Train HiC-SuperNet.

Expects, inside <root_dir>/data (root_dir is set in Arg_Parser.py):
    hicarn_10kb40kb_c40_s40_b201_nonpool_train.npz
    hicarn_10kb40kb_c40_s40_b201_nonpool_valid.npz

Example:
    python HiCSuperNet_Train.py
    python HiCSuperNet_Train.py -e 100 -b 16 -lr 1e-3 --augment

Checkpoints are written to checkpoints/HiCSuperNet/:
    <date>_bestV_..._HiCSuperNet.pytorch   lowest validation loss  (use this for prediction)
    <date>_finalV_..._HiCSuperNet.pytorch  weights after the last epoch
"""

import argparse
import math
import os
import time
from math import log10

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

from Arg_Parser import root_dir
from Models.HiCSuperNet_model import Generator, count_params
from Utils.SSIM import ssim
from Utils.losses import improved_loss, pcc


def parse_args():
    p = argparse.ArgumentParser(description='Train HiC-SuperNet (DiCARN-style pipeline)')
    p.add_argument('-e', '--epochs', type=int, default=100)
    p.add_argument('-b', '--batch_size', type=int, default=16)
    p.add_argument('-lr', '--learning_rate', type=float, default=1e-3)
    p.add_argument('-p', '--patience', type=int, default=15, help='early-stopping patience (epochs)')
    p.add_argument('--base_filters', type=int, default=64)
    p.add_argument('--num_blocks', type=int, default=8)
    p.add_argument('--augment', action='store_true', help='random horizontal/vertical flips')
    p.add_argument('--resos', default='10kb40kb')
    p.add_argument('--chunk', type=int, default=40)
    p.add_argument('--stride', type=int, default=40)
    p.add_argument('--bound', type=int, default=201)
    p.add_argument('--pool', default='nonpool')
    p.add_argument('--name', default='HiCSuperNet')
    p.add_argument('--device', default='auto', help='auto | cuda | mps | cpu')
    p.add_argument('--seed', type=int, default=42)
    return p.parse_args()


def pick_device(choice):
    if choice != 'auto':
        return torch.device(choice)
    if torch.cuda.is_available():
        return torch.device('cuda:0')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def load_split(path):
    d = np.load(path, allow_pickle=True)
    x = torch.tensor(d['data'], dtype=torch.float)
    y = torch.tensor(d['target'], dtype=torch.float)
    return x, y


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    start = time.time()

    device = pick_device(args.device)
    print('Device being used:', device)

    data_dir = os.path.join(root_dir, 'data')
    stem = f'hicarn_{args.resos}_c{args.chunk}_s{args.stride}_b{args.bound}_{args.pool}'
    train_x, train_y = load_split(os.path.join(data_dir, f'{stem}_train.npz'))
    valid_x, valid_y = load_split(os.path.join(data_dir, f'{stem}_valid.npz'))
    print(f'Train: {tuple(train_x.shape)}   Valid: {tuple(valid_x.shape)}')

    train_loader = DataLoader(TensorDataset(train_x, train_y), batch_size=args.batch_size,
                              shuffle=True, drop_last=True)
    valid_loader = DataLoader(TensorDataset(valid_x, valid_y), batch_size=args.batch_size, shuffle=False)

    net = Generator(base_filters=args.base_filters, num_blocks=args.num_blocks).to(device)
    print(f'HiC-SuperNet: {args.num_blocks} MSD blocks, {args.base_filters} filters, '
          f'{count_params(net):,} trainable params')

    optimizer = torch.optim.Adam(net.parameters(), lr=args.learning_rate)
    # cosine decay from lr to lr*1e-6 over the run, stepped once per epoch
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs,
                                                           eta_min=args.learning_rate * 1e-6)

    out_dir = os.path.join('checkpoints', args.name)
    os.makedirs(out_dir, exist_ok=True)
    datestr = time.strftime('%m_%d_%H_%M')
    tag = f'{args.resos}_c{args.chunk}_s{args.stride}_b{args.bound}_{args.pool}_{args.name}'
    best_ckpt = os.path.join(out_dir, f'{datestr}_bestV_{tag}.pytorch')
    final_ckpt = os.path.join(out_dir, f'{datestr}_finalV_{tag}.pytorch')

    history = {k: [] for k in ['train_loss', 'valid_loss', 'ssim', 'psnr', 'mse', 'mae', 'pcc']}
    best_valid, no_improve = float('inf'), 0

    for epoch in range(1, args.epochs + 1):
        # ---- train ----
        net.train()
        run_loss, n = 0.0, 0
        bar = tqdm(train_loader, desc=f'[{epoch}/{args.epochs}] train', leave=False)
        for lr_img, hr_img in bar:
            lr_img, hr_img = lr_img.to(device), hr_img.to(device)
            if args.augment:
                if torch.rand(1).item() < 0.5:
                    lr_img, hr_img = lr_img.flip(-1), hr_img.flip(-1)
                if torch.rand(1).item() < 0.5:
                    lr_img, hr_img = lr_img.flip(-2), hr_img.flip(-2)
            optimizer.zero_grad()
            loss = improved_loss(net(lr_img), hr_img)
            loss.backward()
            optimizer.step()
            run_loss += loss.item() * lr_img.size(0)
            n += lr_img.size(0)
            bar.set_postfix(loss=f'{run_loss / n:.5f}')
        scheduler.step()
        train_loss = run_loss / n

        # ---- validate ----
        net.eval()
        sums = dict(loss=0.0, mse=0.0, mae=0.0, ssim=0.0)
        preds, targets = [], []
        nv = 0
        with torch.no_grad():
            for lr_img, hr_img in tqdm(valid_loader, desc=f'[{epoch}/{args.epochs}] valid', leave=False):
                lr_img, hr_img = lr_img.to(device), hr_img.to(device)
                sr = net(lr_img)
                bs = lr_img.size(0)
                sums['loss'] += improved_loss(sr, hr_img).item() * bs
                sums['mse'] += ((sr - hr_img) ** 2).mean().item() * bs
                sums['mae'] += (sr - hr_img).abs().mean().item() * bs
                sums['ssim'] += ssim(sr[:, 0:1], hr_img[:, 0:1]).item() * bs
                preds.append(sr.cpu())
                targets.append(hr_img.cpu())
                nv += bs
        v = {k: s / nv for k, s in sums.items()}
        v['psnr'] = 10 * log10(1 / max(v['mse'], 1e-12))
        v['pcc'] = pcc(torch.cat(preds), torch.cat(targets)).item()

        for k, val in [('train_loss', train_loss), ('valid_loss', v['loss']), ('ssim', v['ssim']),
                       ('psnr', v['psnr']), ('mse', v['mse']), ('mae', v['mae']), ('pcc', v['pcc'])]:
            history[k].append(val)

        print(f"[{epoch}/{args.epochs}] train {train_loss:.5f} | valid {v['loss']:.5f} | "
              f"SSIM {v['ssim']:.4f} PSNR {v['psnr']:.2f} MSE {v['mse']:.6f} "
              f"MAE {v['mae']:.6f} PCC {v['pcc']:.4f} | lr {scheduler.get_last_lr()[0]:.2e}")

        if v['loss'] < best_valid:
            best_valid, no_improve = v['loss'], 0
            torch.save(net.state_dict(), best_ckpt)
            print(f'  best model saved -> {best_ckpt}')
        else:
            no_improve += 1
            if no_improve >= args.patience:
                print(f'  early stopping: no improvement for {args.patience} epochs')
                break

    torch.save(net.state_dict(), final_ckpt)

    tracker = os.path.join('score_tracker', args.name)
    os.makedirs(tracker, exist_ok=True)
    for k, vals in history.items():
        np.savetxt(os.path.join(tracker, f'valid_{k}_{args.name}.txt'), np.array(vals), delimiter=',')

    mins, secs = divmod(time.time() - start, 60)
    hrs, mins = divmod(mins, 60)
    print(f'\nBest checkpoint : {best_ckpt}')
    print(f'Final checkpoint: {final_ckpt}')
    print(f'Scores saved to : {tracker}')
    print(f'Total training time: {int(hrs):02d}:{int(mins):02d}:{secs:05.2f}')


if __name__ == '__main__':
    main()
