import torch, os, sys, cv2
import torch.nn as nn
import functools
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as func

import numpy as np 
import torch, argparse, pdb
def LoG(img):
	weight = [
		[0, 0, 1, 0, 0],
		[0, 1, 2, 1, 0],
		[1, 2, -16, 2, 1],
		[0, 1, 2, 1, 0],
		[0, 0, 1, 0, 0]
	]
	weight = np.array(weight)

	weight_np = np.zeros((1, 1, 5, 5))
	weight_np[0, 0, :, :] = weight
	weight_np = np.repeat(weight_np, img.shape[1], axis=1)
	weight_np = np.repeat(weight_np, img.shape[0], axis=0)

	weight = torch.from_numpy(weight_np).type(torch.FloatTensor).to('cuda:0')

	return func.conv2d(img, weight, padding=1)

def HFEN(output, target):
	return torch.sum(torch.pow(LoG(output) - LoG(target), 2)) / torch.sum(torch.pow(LoG(target), 2))

def L_1(output,target):
    return torch.sum(torch.abs(output-target))/torch.numel(output)

def loss_func(output,pre_output,target,pre_target):
    ls=L_1(output,target)
    lg=HFEN(output,target)
    lt=L_1(pre_output,pre_target)
    return 0.8*ls + 0.1*lg+ 0.1*lt
