# -*- coding: utf-8 -*-
"""
Created on Fri Nov  6 15:04:31 2020

@author: 16534
"""
import re
import pytesseract
from PIL import Image

# 读取图片
im = Image.open('sentence.png')
# 识别文字
string = pytesseract.image_to_string(im,config='digits')

# 读取图片
im2 = Image.open('aaa.jpg')
# 识别文字
string2 = pytesseract.image_to_string(im2,config='digits')
string2=re.sub('[^0-9]+','',string2)  

print(string)

