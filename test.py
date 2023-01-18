
import torch
import os
import sys
import cv2
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import argparse


from model_chn import *
from data_chn import *
from losses_chn import *

# 载入检查点，初始化神经网络模型的参数
def load_checkpoint(filename):
	checkpoint = torch.load(filename)
	model = Auto_Decoder(11)
	model.to('cuda:0')
	optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.99))
	model.load_state_dict(checkpoint['state_dict'])
	optimizer.load_state_dict(checkpoint['optimizer'])

	return model, optimizer


if __name__ == '__main__':
    
    # 从命令行读取参数，具体参数含义请使用help查看
    parser = argparse.ArgumentParser(description='The homework of CG')
    parser.add_argument('--test_dir', type=str, help='Test data directory')
    parser.add_argument('--output_dir', type=str, help='Directory to save output')
    parser.add_argument('--checkpoint', type=str, help='Checkpoint to load')
    parser.add_argument('--width', type=int, help='The width of test data')
    parser.add_argument('--height', type=int, help='The height of test data')

    args = parser.parse_args()

    model, optimizer = load_checkpoint(args.checkpoint)

    size = (args.width, args.height)
    width = size[0]
    height = size[1]

    set_data = data_set(args.test_dir, size)
    loader_data = DataLoader(set_data, batch_size=1, num_workers=0, shuffle=False)
    
    # 遍历dataloader中的每一份测试数据
    for i, item in enumerate(loader_data):
        root=args.output_dir
        path='seq_'+str(i+1)
        out= os.path.join(root,path)
        os.mkdir(out)
        image_data = item['data']
        image_target = item['target']
        
        #对每一份测试数据进行测试
        for j in range(0, 7):
            
            frame_data = image_data[:, j, :, :, :]
            frame_target = image_target[:, j, :, :, :]

            final_inp = {
				'data': frame_data,
				'target': frame_target
			}

            model.set_input(final_inp)
            if j == 0:
                model.reset_hidden()

            output = model()
            
            # 得到全局光照效果图片
            ray = final_inp['target'].clone()
            ray = torch.squeeze(ray, dim=0)
            ray = ray[:3, :, :]
            ray = ray.permute((1, 2, 0))
            ray = ray.cpu().numpy()
            ray *= 255.0
            # 得到由神经网络经过降噪处理的图片
            output = torch.squeeze(output.detach(), dim=0)
            output = output.permute((1, 2, 0))
            output = output.cpu().numpy()
            output *= 255.0

            #得到噪声输入
            original = final_inp['data']
            original= torch.squeeze(original.detach(), dim=0) * 255.0
            original = original.permute((1, 2, 0))
            original = original.cpu().numpy()
            
            # 输出图片的格式：
            # 噪声输入  降噪输出  全局光照效果
            final=cv2.imread("./example.png")
            final = cv2.resize(final, (width*3,height))
            final[:, :width, :] = original[:, :, :3]
            final[:, width:width * 2, :] = output
            final[:, width * 2:width * 3, :] = ray
            image=str(j+1)+'.png'
            out_image=os.path.join(out,image)
            cv2.imwrite(out_image, final)