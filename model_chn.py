'''
本次实验在代码复现阶段参考了部分github上的开源项目，链接如下:
https://github.com/yuyingyeh/rdae，
https://github.com/AakashKT/pytorch-recurrent-ae-siggraph17，
以上代码仅用于参考复现思路以及pytorch部分内置函数的学习，与本次实验复现出的代码存在较大差异，并不存在单纯复制粘贴和抄袭等不当行为。 

'''
import torch
from torch import nn
# 定义神经网络中Recurrent_Block结构
class Recurrent_Block(nn.Module):
    def __init__(self,channel):
        super(Recurrent_Block, self).__init__()
        self.hidden=None
        self.state_1=nn.Sequential(
            nn.Conv2d(channel,channel,3,padding=1),
            nn.LeakyReLU(negative_slope=0.1)
        )
        self.state_2=nn.Sequential(
            nn.Conv2d(2*channel, channel , 3,padding=1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(channel, channel,3,padding=1),
            nn.LeakyReLU(negative_slope=0.1)
        )
    def forward(self,input):
        op1=self.state_1(input)
        op2=self.state_2(torch.cat((op1,self.hidden),dim=1))    #当前数据与隐藏层数据共同输入神经网络
        self.hidden=op2
        return op2
    def reset_hidden(self,size):     #将self.hidden重置成为全0的4维张量，尺寸与该block输出的尺寸相符       
        self.hidden=torch.zeros(*size).to('cuda:0')
        return 

# 定义降噪器中的编码阶段
class Encode_state(nn.Module):
    def __init__(self,input_nc,output_nc):
        super(Encode_state, self).__init__()
        self.input_nc=input_nc
        self.output_nc=output_nc
        self.model=nn.Sequential(
            nn.Conv2d(input_nc, output_nc, 3,padding=1),
            nn.LeakyReLU(negative_slope=0.1),
        )
        self.block=Recurrent_Block(output_nc)
        self.pool=nn.MaxPool2d(2)
    def forward(self,input):
        op1=self.model(input)
        op2=self.block(op1)
        op3=self.pool(op2)
        return op3
    def reset_hidden(self,size,layer):      #调用Recurrent_Block中提供的接口函数reset_hidden来初始化隐藏层
        tmp=size.copy()
        tmp[1]=self.output_nc
        tmp[2]=int(size[2]/2**layer)
        tmp[3]=int(size[3]/2**layer)
        self.block.reset_hidden(tmp)
        return

#定义降噪器中的解码阶段
class Decode_state(nn.Module):
    def __init__(self,input_nc,output_nc):
        super(Decode_state, self).__init__()
        self.model=nn.Sequential(
            nn.Upsample(scale_factor=2,mode='nearest'),
            nn.Conv2d(2*input_nc, output_nc, 3,padding=1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(output_nc, output_nc, 3, padding=1),
            nn.LeakyReLU(negative_slope=0.1)
        )
    def forward(self,input):
        output=self.model(input)
        return output

#定义降噪器中的Bottle_neck阶段
class Bottle_neck(nn.Module):
    def __init__(self,channel):
        super(Bottle_neck, self).__init__()
        self.channel=channel
        self.l1=nn.Conv2d(channel, channel, 3,padding=1)
        self.l2=nn.LeakyReLU(negative_slope=0.1)
        self.block=Recurrent_Block(channel)
    def forward(self,input):
        op1=self.l1(input)
        op2=self.l2(op1)
        op3=self.block(op2)
        return op3
    def reset_hidden(self,size,layer):      #调用Recurrent_Block中提供的接口函数reset_hidden来初始化隐藏层
        tmp=size.copy()
        tmp[1]=self.channel
        tmp[2]=int(size[2]/2**layer)
        tmp[3]=int(size[3]/2**layer)
        self.block.reset_hidden(tmp)
        return

# 定义整个神经网络，神经网络由上述三个阶段组成，分别是encode_state bottle_neck decode_state
class Auto_Decoder(nn.Module):
    def __init__(self,input_nc):
        super(Auto_Decoder,self).__init__()
        #整个神经网络包含5个编码阶段，每经过一个阶段，分辨率下降
        self.Encode_1=Encode_state(input_nc, 32)
        self.Encode_2=Encode_state(32, 43)
        self.Encode_3=Encode_state(43, 57)
        self.Encode_4=Encode_state(57, 76)
        self.Encode_5=Encode_state(76, 101)

        self.Bottle_neck=Bottle_neck(101)

        #整个神经网络包含5个解码阶段，每经过一个阶段，分辨率上升
        self.Decode_1=Decode_state(101, 76)
        self.Decode_2=Decode_state(76, 57)
        self.Decode_3=Decode_state(57, 43)
        self.Decode_4=Decode_state(43, 32)
        self.Decode_5=Decode_state(32, 3)
    
    def set_input(self,inp):    #设置神经网络的输入
        self.inp=inp['data']
        self.size=list(self.inp.size())

    def forward(self):
        op1=self.Encode_1(self.inp)
        op2=self.Encode_2(op1)
        op3=self.Encode_3(op2)
        op4=self.Encode_4(op3)
        op5=self.Encode_5(op4)

        mid=self.Bottle_neck(op5)
        
        po1=self.Decode_1(torch.cat((mid,op5),dim=1))
        po2=self.Decode_2(torch.cat((po1,op4),dim=1))
        po3=self.Decode_3(torch.cat((po2,op3),dim=1))
        po4=self.Decode_4(torch.cat((po3,op2),dim=1))
        po5=self.Decode_5(torch.cat((po4,op1),dim=1))

        return po5
    def reset_hidden(self): #利用encode_state bottle_neck提供的接口函数初始化隐藏层
        tmp=self.size.copy()
        self.Encode_1.reset_hidden(tmp,0)
        self.Encode_2.reset_hidden(tmp,1)
        self.Encode_3.reset_hidden(tmp,2)
        self.Encode_4.reset_hidden(tmp,3)
        self.Encode_5.reset_hidden(tmp,4)
        self.Bottle_neck.reset_hidden(tmp, 5)
        return

