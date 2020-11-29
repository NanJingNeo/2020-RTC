import cv2
# import numpy as np
import pytesseract
import re

#改它就好
PATH1='TX_Proftalk_100'

#获得视频
PATH0='./video/'

PATH2='.mp4'
videoCapture = cv2.VideoCapture(PATH0+PATH1+PATH2)

frames_num=videoCapture.get(7)
print("视频共"+str(int(frames_num))+"帧")
# 读帧
success, frame = videoCapture.read()

#帧的序号
name_cnt = 0
text1=0

#存数据的表
ocr_=list()         #保留重复序号
ocr_count_=list()    #除去重复序号
multi_=list()     #记录100*n的所在帧数
multi_100_=list()    #记录读取到的100*n
#读取的套路

while success:
    
    #读取文本
    text = pytesseract.image_to_string(frame,config='digits')
    text=re.sub('[^0-9]+','',text)
    if (text ==''):
        text = text1
    ocr_.append(text)
    if (text != text1):
       ocr_count_.append(text)
    if ((int(text)%100) == 0) and (text != text1):         # 判断变量是否为 python 
       img_name = "./picture/"+PATH1+'_'+str(name_cnt)+".png"
       multi_.append(name_cnt)
       multi_100_.append(text)
       #保存图片
       cv2.imwrite(img_name, frame)  
    print("视频共"+str(int(frames_num))+"帧,""第"+str(name_cnt)+"帧已识别")            
    name_cnt = name_cnt + 1
    text1=text
    
    # 获取下一帧
    success, frame = videoCapture.read() 
    
with open('./txt/'+PATH1+'ocr.txt', 'w+', encoding='utf-8') as f:
# ocr存储
    for ocr in ocr_:
        f.write(ocr + "\n")
        
with open('./txt/'+PATH1+'ocr_count.txt', 'w+', encoding='utf-8') as f:
# ocr存储
    f.write("共"+str(len(ocr_count_)) +"帧"+ "\n")
    for ocr_count in ocr_count_:
        f.write(ocr_count + "\n")

with open('./txt/'+PATH1+'multi.txt', 'w+', encoding='utf-8') as f:
# 将各个图片的路径写入text.txt文件当中
    for multi in multi_:
        f.write(str(multi) + '\n')
        
with open('./txt/'+PATH1+'multi_100.txt', 'w+', encoding='utf-8') as f:
# 将各个图片的路径写入text.txt文件当中
    for multi_100 in multi_100_:
        f.write(str(multi_100) + '\n')
        


