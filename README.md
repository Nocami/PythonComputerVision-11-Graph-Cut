# PythonComputerVision-11--
图像分割是将一幅图像分割成有意义区域的过程。区域可以是图像的前景与背景或者单个对象。这些区域可以利用诸如颜色、边线或近邻相似性等特征构建。本文，将介绍一些不同的分割技术
# 图像分割

我们先看一下帝国大厦的这张原图：  
![image](https://github.com/Nocami/PythonComputerVision-11--/blob/master/image/empire.jpg)  
图像分割后的图片如下：  
![image](https://github.com/Nocami/PythonComputerVision-11--/blob/master/image/1.jpg)  

源码如下：  
~~~python
# -*- coding: utf-8 -*-

from scipy.misc import imresize
from PCV.tools import graphcut
from PIL import Image
from numpy import *
from pylab import *

im = array(Image.open("empire.jpg"))
im = imresize(im, 0.07)
size = im.shape[:2]
print ("OK!!")

# add two rectangular training regions
labels = zeros(size)
labels[3:18, 3:18] = -1
labels[-18:-3, -18:-3] = 1
print ("OK!!")


# create graph
g = graphcut.build_bayes_graph(im, labels, kappa=1)

# cut the graph
res = graphcut.cut_graph(g, size)
print ("OK!!")


figure()
graphcut.show_labeling(im, labels)

figure()
imshow(res)
gray()
axis('off')

show()
~~~
