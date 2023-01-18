import math
 
import numpy as np
from PIL import Image
from scipy.signal import convolve2d
 
 
# target:目标图像  ref:参考图像 
def PSNR(target, ref):
    target = np.array(target, dtype=np.float32)
    ref = np.array(ref, dtype=np.float32)
 
    if target.shape != ref.shape:
        raise ValueError('输入图像的大小应该一致！')
 
    diff = ref - target
    diff = diff.flatten('C')
 
    rmse = math.sqrt( np.mean(diff ** 2.) )
    psnr = 20 * math.log10(np.max(target) / rmse)
 
    return psnr
 
 
def SSIM(target, ref, K1=0.01, K2=0.03, gaussian_kernel_sigma = 1.5, gaussian_kernel_width = 11, L=255):
    # 高斯核，方差为1.5，滑窗为11*11
    gaussian_kernel = np.zeros((gaussian_kernel_width,gaussian_kernel_width))
    for i in range(gaussian_kernel_width):
        for j in range(gaussian_kernel_width):
            gaussian_kernel[i,j] = (1 / (2 * math.pi * (gaussian_kernel_sigma ** 2))) * math.exp(-(((i-5) ** 2)+((j - 5) ** 2)) / (2 * (gaussian_kernel_sigma ** 2)))
 
    target = np.array(target, dtype=np.float32)
    ref = np.array(ref, dtype=np.float32)
    if target.shape != ref.shape:
        raise ValueError('输入图像的大小应该一致！')
 
    target_window = convolve2d(target, np.rot90(gaussian_kernel, 2), mode='valid')
    ref_window = convolve2d(ref, np.rot90(gaussian_kernel, 2), mode='valid')
 
    mu1_sq = target_window * target_window
    mu2_sq = ref_window * ref_window
    mu1_mu2 = target_window * ref_window
 
    sigma1_sq = convolve2d(target * target, np.rot90(gaussian_kernel, 2), mode='valid') - mu1_sq
    sigma2_sq = convolve2d(ref * ref, np.rot90(gaussian_kernel, 2), mode='valid') - mu2_sq
    sigma12 = convolve2d(target * ref, np.rot90(gaussian_kernel, 2), mode='valid') - mu1_mu2
 
    C1 = (K1*L)**2
    C2 = (K2*L)**2
    ssim_array = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    ssim = np.mean(np.mean(ssim_array))
 
    return ssim
 
 
if __name__ == '__main__':
    target = Image.open('./C.png').convert('L')
    ref = Image.open('./B.png').convert('L')
 
    psnr = PSNR(target, ref)
    print('PSNR为:{}'.format(psnr))
 
    ssim = SSIM(target, ref)
    print('SSIM为:{}'.format(ssim))