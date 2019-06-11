# PythonComputerVision-11--
图像分割是将一幅图像分割成有意义区域的过程。区域可以是图像的前景与背景或者单个对象。这些区域可以利用诸如颜色、边线或近邻相似性等特征构建。本文，将介绍一些不同的分割技术
# 一.图割（Graph Cut）
图切是将一个有向图分割成两个互不相交的集合，可以用来解决很多计算机视觉方面的问题，诸如立体深度重建、图像拼接和图像分割。从图像像素和像素的邻近创建一个图并引入一个能量或“代价”函数，即有可能利用图割方法将图像分割成两个或多个区域。其基本思想是，相似且彼此相近的像素应该划分到同一区域。  
图割C（C是图中所有边的集合）的代价函数定义为所有割的边的权重求和相加。  
**寻找最小割**（minimum cut 或min cut）等同于在源点和汇点间寻找**最大流**（maxmum flow或max flow）。在图割例子中，要用到"python-graph"工具包，可以在 http://code.google.com/p/python-graph 下载。  
### 简单例子：
我们先来给出一个用python-graph工具包计算一副较小图的最大流/最小割的简单例子：  
~~~python
from pygraph.classes.digraph import digraph
from pygraph.algorithms.minmax import maximum_flow

gr = digraph()
gr.add_nodes([0,1,2,3])
gr.add_edge((0,1), wt=4)
gr.add_edge((1,2), wt=3)
gr.add_edge((2,3), wt=5)
gr.add_edge((0,2), wt=3)
gr.add_edge((1,3), wt=4)
flows,cuts = maximum_flow(gr, 0, 3)
print ('flow is:' , flows)
print ('cut is:' , cuts)
~~~  
首先，创建有4个节点的有向图，4个节点的索引分别为0 1 2 3，然后用add_edge()增添边并为每条边指定特定的权重。边的权重用来衡量边的最大流容量。以节点0为源点，3为汇点，计算最大流。结果如下：  
![image](https://github.com/Nocami/PythonComputerVision-11--/blob/master/image/4.jpg)  
结果包含了流穿过每条边和每个节点的标记：0是包含图源点的部分，1是与汇点相连的节点。这个割是最小的。
### 从图像创建图
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
