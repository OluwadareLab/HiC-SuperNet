"""
Predict with HiC-SuperNet.

Example:
    python Predict_HiCSuperNet.py -m HiCSuperNet -lr 40kb \
        -ckpt checkpoints/HiCSuperNet/<date>_bestV_10kb40kb_c40_s40_b201_nonpool_HiCSuperNet.pytorch \
        -f hicarn_10kb40kb_c40_s40_b201_nonpool_GM12878_test.npz -c GM12878_HiCSuperNet
"""
import sys
import os
import time
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from math import log10
import torch

from Models.HiCSuperNet_model import Generator

from Utils.SSIM import ssim
from Utils.GenomeDISCO import compute_reproducibility
from Utils.io import spreadM, together
from Arg_Parser import *


def dataloader(data, batch_size=64):
	inputs = torch.tensor(data['data'], dtype=torch.float)
	target = torch.tensor(data['target'], dtype=torch.float)
	inds = torch.tensor(data['inds'], dtype=torch.long)
	dataset = TensorDataset(inputs, target, inds)
	loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
	return loader
	
def get_chr_nums(data):
	inds = torch.tensor(data['inds'], dtype=torch.long)
	chr_nums = sorted(list(np.unique(inds[:, 0])))
	return chr_nums


def data_info(data):
	indices = data['inds']
	compacts = data['compacts'][()]
	sizes = data['sizes'][()]
	return indices, compacts, sizes


get_digit = lambda x: int(''.join(list(filter(str.isdigit, x))))


def filename_parser(filename):
	info_str = filename.split('.')[0].split('_')[2:-1]
	chunk = get_digit(info_str[0])
	stride = get_digit(info_str[1])
	bound = get_digit(info_str[2])
	scale = 1 if info_str[3] == 'nonpool' else get_digit(info_str[3])
	return chunk, stride, bound, scale


def hicarn_predictor(model, hicarn_loader, ckpt_file, device, data_file, base_filters=64, num_blocks=8):
	deepmodel = Generator(base_filters=base_filters, num_blocks=num_blocks).to(device)
	if not os.path.isfile(ckpt_file):
		ckpt_file = f'save/{ckpt_file}'
	deepmodel.load_state_dict(torch.load(ckpt_file, map_location=device))
	print(f'Loading checkpoint file from "{ckpt_file}"')

	result_data = []
	result_inds = []

	chr_nums = get_chr_nums(data_file)
	print("Chromosomes: ", chr_nums)

	# per-sample scores, grouped by each sample's own chromosome
	# (a batch can contain samples from two chromosomes)
	per_chr = {int(c): {'ssim': [], 'mse': [], 'gd': []} for c in chr_nums}

	deepmodel.eval()
	with torch.no_grad():
		for batch in tqdm(hicarn_loader, desc='HiC-SuperNet Model Testing...: '):
			lr, hr, inds = batch
			lr = lr.to(device)
			hr = hr.to(device)
			out = deepmodel(lr)

			mses = ((out - hr) ** 2).mean(dim=(1, 2, 3)).cpu()
			ssims = ssim(out[:, 0:1, :, :], hr[:, 0:1, :, :], size_average=False).cpu()
			out_np = out.cpu().numpy()
			hr_np = hr.cpu().numpy()

			for k in range(out_np.shape[0]):
				c = int(inds[k][0])
				per_chr[c]['mse'].append(mses[k].item())
				per_chr[c]['ssim'].append(ssims[k].item())
				per_chr[c]['gd'].append(compute_reproducibility(out_np[k, 0], hr_np[k, 0], transition=True))

			result_data.append(out_np)
			result_inds.append(inds.numpy())
	result_data = np.concatenate(result_data, axis=0)
	result_inds = np.concatenate(result_inds, axis=0)

	mean_ssims, mean_mses, mean_psnrs, mean_gds = [], [], [], []
	for c, sc in per_chr.items():
		if not sc['mse']:
			continue
		m_ssim = float(np.mean(sc['ssim']))
		m_mse = float(np.mean(sc['mse']))
		m_psnr = 10 * log10(1 / max(m_mse, 1e-12))
		m_gd = float(np.mean(sc['gd']))
		mean_ssims.append(m_ssim); mean_mses.append(m_mse); mean_psnrs.append(m_psnr); mean_gds.append(m_gd)

		print("\n")
		print("Chr", c, "SSIM: ", round(m_ssim, 4))
		print("Chr", c, "MSE: ", round(m_mse, 4))
		print("Chr", c, "PSNR: ", round(m_psnr, 4))
		print("Chr", c, "GenomeDISCO: ", round(m_gd, 4))

	print("\n")
	print("___________________________________________")
	print("Means across chromosomes")
	print("SSIM: ", round(float(np.mean(mean_ssims)), 4))
	print("MSE: ", round(float(np.mean(mean_mses)), 4))
	print("PSNR: ", round(float(np.mean(mean_psnrs)), 4))
	print("GenomeDISCO: ", round(float(np.mean(mean_gds)), 4))
	print("___________________________________________")
	print("\n")

	hicarn_hics = together(result_data, result_inds, tag='Reconstructing: ')
	return hicarn_hics        	
	

def save_data(carn, compact, size, file):
	hicarn = spreadM(carn, compact, size, convert_int=False, verbose=True)
	np.savez_compressed(file, hicarn=hicarn, compact=compact)
	print('Saving file:', file)


if __name__ == '__main__':
	args = data_predict_parser().parse_args(sys.argv[1:])
	cell_line = args.cell_line
	low_res = args.low_res
	ckpt_file = args.checkpoint
	cuda = args.cuda
	model = args.model
	HiCARN_file = args.file_name
	print('NOTE: rebuilding full chromosome maps needs a lot of RAM for large test sets.')

	in_dir = os.path.join(root_dir, 'data')
	out_dir = os.path.join(root_dir, 'predict', cell_line)
	mkdir(out_dir)

	files = [f for f in os.listdir(in_dir) if f.find(low_res) >= 0]

	chunk, stride, bound, scale = filename_parser(HiCARN_file)

	if args.device != 'auto':
		device = torch.device(args.device)
	elif torch.cuda.is_available() and -1 < cuda < torch.cuda.device_count():
		device = torch.device(f'cuda:{cuda}')
	elif torch.backends.mps.is_available():
		device = torch.device('mps')
	else:
		device = torch.device('cpu')
	print(f'Using device: {device}')


	start = time.time()
	print(f'Loading data[GM12878]: {HiCARN_file}')
	hc_data = os.path.join(in_dir, HiCARN_file)
	hicarn_data = np.load(os.path.join(in_dir, HiCARN_file), allow_pickle=True)
	hicarn_loader = dataloader(hicarn_data)

	print("Input Test Data: ", hc_data)
	
	print("Checkpoint in use: ", ckpt_file)


	indices, compacts, sizes = data_info(hicarn_data)

	hicarn_hics = hicarn_predictor(model, hicarn_loader, ckpt_file, device, hicarn_data,
	                               base_filters=args.base_filters, num_blocks=args.num_blocks)


	def save_data_n(key):
		file = os.path.join(out_dir, f'predict_chr{key}_{low_res}.npz')
		save_data(hicarn_hics[key], compacts[key], sizes[key], file)



	# Save one chromosome at a time (works on macOS; the original multiprocessing pool fails silently there)
	for key in compacts.keys():
		save_data_n(key)
	print(f'All data saved. Running cost is {(time.time() - start) / 60:.1f} min.')
