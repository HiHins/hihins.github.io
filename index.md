  机器学习
数据的预处理	Numpy	Pandas
画图可视化	Matplotlib	
神经网络	TensorFlow	Pytorch
1.Numpy学习
导入numpy模块：import numpy as np
array = np.array([[1, 2, 3],
                 [2, 3, 4]])//创建矩阵，两层中括号
也可以为矩阵中的数据指定格式：
array = np.array([[1, 2,3],
                  [2, 3, 4]],dtype=float)
创建全是0的3行4列矩阵：
array=np.zeros((3,4))//双层括号
创建全是1的矩阵：
array=np.ones((3,4))
创建数组：
array=np.arange(m,n,i)//从m到n-1的i步长
或者：array=np.arange(t)//从0到t-1
转换成m行n列矩阵：
a=array.reshape(m,n)
矩阵的加减法：
a=np.array([10,20,30,40])
b=np.arange(4)//0-3
c=a+b
求矩阵每个值的三角函数值：
d=np.tan(a)
求矩阵每个值的布尔值：
print(b>3)//[True True True False]
print(b==3)//[False False False True]
矩阵运算：
a=np.array([[1,1],[0,1]])
b=np.arange(4).reshape(2,2)
c=a*b//普通乘法
c_dot=np.dot(a,b)//矩阵乘法
d_dot=a.dot(b)//和上面相同
随机产生矩阵：
a=np.random.random((2,4))//随机产生两行四列矩阵
对矩阵运用：
np.sum(a)//求矩阵各数之和
np.min(a)//求矩阵中最小值
np.max(a)//求矩阵中最大值
np.sum(a,axis=0)//在列中求和
np.sum(a,axis=1)//在行中求和
np.argmin(a)//求最小值的索引
np.cumsum(a)//累加
np.diff(a)//累差
np.sort(a)//逐行排序
a.T//矩阵转置
np.clip(a,5,9)//将矩阵中的小于5的改成5，大于9的改成9
np.mean(a,axis=1)//对行求平均值
np.mean(a,axis=0)//对列求平均值
a[1][1]或者a[1,1]//用索引求矩阵中的位置值
np.vstack((a,b))//矩阵上下合并  vertical垂直的
np.hstack((a,b))//矩阵左右合并  horizontal水平的
分割array：
