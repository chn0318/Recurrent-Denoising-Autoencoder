
import torch
import os, sys
import cv2
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np 
import argparse
from torch.utils.tensorboard import SummaryWriter


from model_chn import *
from data_chn import *
from losses_chn import *
def get_temporal_data(output, target):      #获得时域上的损失函数值
	final_output = output.clone()
	final_target = target.clone()
	final_output.fill_(0)
	final_target.fill_(0)

	for i in range(1, 7):
		final_output[:, i, :, :, :] = output[:, i, :, :] - output[:, i-1, :, :]
		final_target[:, i, :, :, :] = target[:, i, :, :] - target[:, i-1, :, :]

	return final_output, final_target
def train_frame(model,dic):                #对连续7帧图像进行训练
    #初始化输出、目标值
    output_final=dic['target'].clone()
    target_final=dic['target'].clone()
    target_final.fill_(0)
    output_final.fill_(0)
    
    input_seq=dic['data']
    target_seq=dic['target']
    loss_final=0
    
    #将图像输入神经网络
    for i in range(0,7):                    #对单个一帧图片进行训练
        input_image=input_seq[:,i,:,:,:]
        target_image=target_seq[:,i,:,:,:]
        input_data={
            'data':input_image,
            'target': target_image
        }
        model.set_input(input_data)
        if i==0:
            model.reset_hidden()
        
        #从神经网络中获得该帧图像的输出
        output=model()
        output_final[:,i,:,:,:]=output
        target_final[:,i,:,:,:]=target_image
    temporal_output, temporal_target = get_temporal_data(output_final, target_final)
   
   
    # 计算损失值
    for j in range(0,7):
        output_image=output_final[:,j,:,:,:]
        target_image=target_final[:,j,:,:,:]
        pre_output = temporal_output[:, j, :, :, :]
        pre_target = temporal_target[:, j, :, :, :]
        l=loss_func(output_image,pre_output,target_image,pre_target)
        loss_final+=l
    return loss_final


def save_checkpoint(state, filename):
	torch.save(state, filename);

if __name__ == '__main__':

    #从命令行读取参数
    parser = argparse.ArgumentParser(description='The homework of CG')
    parser.add_argument('--name', type=str, help='Experiment Name')
    parser.add_argument('--data_dir', type=str, help='Data directory')
    parser.add_argument('--save_dir', type=str, help='Model chekpoint saving directory')
    parser.add_argument('--width',type=int, help='The width of training data ')
    parser.add_argument('--height',type=int, help='The height of training data')
    parser.add_argument('--epochs', type=int, help='Number of epochs to train')

    args = parser.parse_args()
    
    #启动 TensorBoard
    writer = SummaryWriter("./logs_train")
    set_data = data_set(args.data_dir, (args.width, args.height))
    print(set_data.__len__())
    loader_data = DataLoader(set_data, batch_size=1, num_workers=0, shuffle=True)

    model=Auto_Decoder(11)
    model.to('cuda:0')

    #打印模型结构
    print(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.99))

    for epoch in range(args.epochs):
        print('--------第{}轮训练开始------'.format(epoch+1))
        total_loss=0
        total_loss_num=0
        for i,dic in enumerate(loader_data):               #  该循环中遍历dataLoader中的每一个元素,dic返回一个字典{"data": 五维张量，"target": 五维张量}
            
            #使用pytorch 中内置的优化器进行优化
            optimizer.zero_grad()
            loss_image=train_frame(model,dic)
            loss_image.backward(retain_graph=False)
            optimizer.step()
            
            #训练过程中的输出信息，方便监控
            if i % 50==0:
                print('第{}份训练数据的损失函数值为{}'.format(i+1,loss_image.item()))
            
            total_loss+=loss_image.item()
            total_loss_num+=1
        
        # 计算一轮训练中的总损失函数值，并加入TensorBoard
        total_loss/=total_loss_num  
        writer.add_scalar("total_loss", total_loss,epoch)
        print('第{}轮训练中 total_loss = {}'.format(epoch+1, total_loss))
        sys.stdout.flush()

        #每训练50轮保存一次模型信息
        if epoch % 50 == 0:
            print('第{}轮训练模型已保存!'.format(epoch+1))
            save_checkpoint({
					'epoch': epoch+1,
					'state_dict':model.state_dict(),
					'optimizer':optimizer.state_dict(),
				}, '%s/%s_%s.pt' % (args.save_dir, args.name, epoch+1))
   
    # 保存最后一次训练的模型信息
    print('第{}轮训练模型已保存!'.format(args.epochs+1))
    save_checkpoint({
				'epoch': args.epochs,
				'state_dict':model.state_dict(),
				'optimizer':optimizer.state_dict(),
			}, '%s/%s_%s.pt' % (args.save_dir, args.name, args.epochs))
    writer.close()