import numpy as np
import math
import cv2
# import time
# from PIL import Image

#每次只要改它们就好
PATH1='TX_Proftalk_100'

path='./videoCapture0/Proftalk.mp4'

def psnr(img1, img2):
    img1 = np.float64(img1)
    img2 = np.float64(img2)
    mse = np.mean( (img1 - img2) ** 2 )
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

#作为参考，1和2结果一样

def psnr0(target, ref):
	#将图像格式转为float64
    target_data = np.array(target, dtype=np.float64)
    ref_data = np.array(ref,dtype=np.float64)
    # 直接相减，求差值
    diff = ref_data - target_data
    # 按第三个通道顺序把三维矩阵拉平
    diff = diff.flatten('C')
    # 计算MSE值
    rmse = math.sqrt(np.mean(diff ** 2.))
    # 精度
    eps = np.finfo(np.float64).eps
    if(rmse == 0):
        rmse = eps 
    return 20*math.log10(255.0/rmse)


def ssim(img1, img2):
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())
    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def calculate_ssim(img1, img2):
    '''calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    '''
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1, img2))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')

PATH0='./videoCapture1/'

PATH2='.mp4'

#录屏
path1=PATH0+PATH1+PATH2
videoCapture1 = cv2.VideoCapture(path1)

#原视频

videoCapture0 = cv2.VideoCapture(path)

#初始化两个图像指标
psnrValue=0
ss=0

# frames_num=videoCapture1.get(7)
# print("视频共"+str(int(frames_num))+"帧")


with open('./txt/'+PATH1+'multi.txt', 'r' , encoding='utf-8') as f:
    multi = f.readlines()  #txt中所有字符串读入data
    
        
with open('./txt/'+PATH1+'multi_100.txt', 'r' , encoding='utf-8') as f:
    multi_100 = f.readlines()  #txt中所有字符串读入data
    # for i in range(len(multi_100)):
    #     multi_100[i]=multi_100[i].replace('\n','')        #去除"\n"
    
for i in range(len(multi)):   
    #原视频帧号不重复，直接选就好    
    videoCapture0.set(cv2.CAP_PROP_POS_FRAMES,int(multi_100[i])-10000)  #设置要获取的帧号
    #录屏视频帧号需要从文本里找
    videoCapture1.set(cv2.CAP_PROP_POS_FRAMES,int(multi[i]))  #设置要获取的帧号   
    a,b=videoCapture1.read()  #read方法返回一个布尔值和一个视频帧。若帧读取成功，则返回True
    cv2.imwrite('./pic/'+str(i)+'b.jpg', b)
    A,B=videoCapture0.read()
    cv2.imwrite('./pic/'+str(i)+'B.jpg', B)    
    psnrValue = psnrValue+psnr(B,b)/len(multi)
    ss = ss+calculate_ssim(b, B)/len(multi)
    print("图片共"+str(len(multi))+"组,""第"+str(i)+"组已计算") 

# path='./txt/VQ'+str(time.strftime("%Y%m%d%H%M%S", time.localtime()))+'.txt'
path='./txt/VQ'+PATH1+'.txt'
with open(path, 'w+', encoding='utf-8') as f:
# 将各个图片的路径写入text.txt文件当中
        f.write("视频 "+path+" 平均峰值信噪比为 "+str(psnrValue) + '\n' + "平均结构一致性参数为 " + str(ss))
    
     # cv2.imshow('b', b)    
    
# original = cv2.imread("1.png")      # numpy.adarray
# contrast = cv2.imread("2.png",1)

# img1 = cv2.imread("1.png", 0)
# img2 = cv2.imread("2.png", 0)
# img1 = np.array(img1)
# img2 = np.array(img2)

# original1 = Image.open('1.png')
# contrast1 = Image.open('2.png')



# psnrValue = psnr(original,contrast)
# print(psnrValue)
# ss = calculate_ssim(img1, img2)
# print(ss)