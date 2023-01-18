
import torch, os, sys, cv2
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader

import torchvision.transforms as transforms
import numpy as np 

class data_set(Dataset):
    def __init__(self,input_dir,size):
        super(data_set, self).__init__()
        self.data_path=input_dir                                     #训练数据集根文件夹路径
        self.frame_path_list=sorted(os.listdir(self.data_path))      #7个连续帧组成一组训练数据，所有训练数据组成一个list
        self.width=size[0]
        self.height=size[1]


    def __getitem__(self, index):
        
        #初始化输入数据以及目标
        data = np.zeros((7,self.height, self.width,11), dtype=np.float)
        target = np.zeros((7,self.height, self.width,3), dtype=np.float)
        path=os.path.join(self.data_path,self.frame_path_list[index])
        
        #获得连续7帧图像组成的列表数据结构，并且逐个遍历
        seq_list=sorted(os.listdir(path))
        for i, item in enumerate(seq_list):
            
            #拆分原始数据
            #原始数据格式为：
            # noise    target
            # albedo   normal
            # depth    roughness
            imge = cv2.imread('%s/%s' % (path,item))
            shading = imge[:self.height, :self.width, :]  
            ray_shading = imge[:self.height, self.width:, :]
            albedo = imge[self.height:self.height * 2, :self.width, :]
            normal = imge[self.height:self.height * 2, self.width:, :]
            depth = (imge[self.height * 2:, :self.width, 0] + imge[self.height * 2:, :self.width, 1] \
								+ imge[self.height * 2:, :self.width, 2]) / 3
            roughness = (imge[self.height * 2:, self.width:, 0] + imge[self.height * 2:, self.width:, 1] \
								+ imge[self.height * 2:, self.width:, 2]) / 3
            depth = np.expand_dims(depth, axis=2)
            roughness = np.expand_dims(roughness, axis=2)
            
            #数据归一化处理
            ray_shading = ray_shading.astype(np.float) / 255.0
            shading = shading.astype(np.float) / 255.0
            normal = normal.astype(np.float) / 255.0
            albedo = albedo.astype(np.float) / 255.0
            depth = depth.astype(np.float) / 255.0
            roughness = roughness.astype(np.float) / 255.0

            data[i,:,:,:3]=shading
            data[i,:,:,3:6]=normal
            data[i,:,:,6:9]=albedo
            data[i,:,:,9:10]=depth
            data[i,:,:,10:11]=roughness

            target[i,:,:,:3]=ray_shading

        data=torch.from_numpy(data)
        target=torch.from_numpy(target)

        #data作为4维张量，每个维度代表：第i帧，通道数，高，宽，target同理
        data = data.permute((0, 3, 1, 2))              
        target = target.permute((0, 3, 1, 2))

        #返回一个字典
        return {
            'data': data.type(torch.float).to('cuda:0'),
            'target': target.type(torch.float).to('cuda:0')
        }



    def __len__(self):
        #获得数据集大小
        return len(self.frame_path_list)