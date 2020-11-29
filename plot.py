# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 11:53:34 2020

@author: 16534
"""


import matplotlib.pyplot as plt
import matplotlib
from matplotlib.ticker import FuncFormatter
import numpy as np
from pylab import *

PATH1='TX_Proftalk_100'

mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['axes.unicode_minus'] = False

#y轴
with open('./txt/'+PATH1+'ocr.txt', 'r' , encoding='utf-8') as f:
    multi_100 = f.readlines()  #txt中所有字符串读入data
    
for i in range(len(multi_100)):
    multi_100[i]=int(multi_100[i])-10000

# r11=[0.7822,0.7763,0.7889,0.7837,0.7785,0.7881,0.7815,0.7741,0.7837,0.7852]

#axis轴
# axis=['1st','2nd','3rd','4th','5th','6th','7th','8th','9th','10th']
axis=np.arange(0,len(multi_100),1)

# def to_percent(temp, position):
#     return '%1.00f'%(100*temp) + '%'

#绘图
# plt.figure(figsize=(8,4.5))
zhfont1 = matplotlib.font_manager.FontProperties(fname='C:\Windows\Fonts\simkai.ttf')
plt.xlabel('录屏帧号', fontproperties=zhfont1)
plt.ylabel('视频帧号', fontproperties=zhfont1)
# plt.ylim(0.6,0.95)
plt.plot(axis, multi_100 , linewidth=1.0, linestyle='-',label='frame_num')

# for a, b ,c in zip(axis1, r11, r12):
#     plt.text(a, b, b, ha='center', va='bottom',fontsize=8)
#     plt.text(a, c, c, ha='center', va='top',fontsize=8)   
plt.legend()
# plt.gca().yaxis.set_major_formatter(FuncFormatter(to_percent))
plt.grid()  # 生成网格
plt.savefig('./plot/PATH1'+'.png',dpi=200)
plt.show()