import tensorflow.compat.v1 as tf
import numpy as np
import pprint
tf.set_random_seed(777)    # for reproducibility

pp = pprint.PrettyPrinter(indent=4)
sess = tf.InteractiveSession()

## Simple Array
t = np.array([0., 1., 2., 3., 4., 5., 6.])
pp.pprint(t) 
print(t)
print(t.ndim) # rank
print(t.shape) # shape
print(t[0], t[1], t[-1])
print(t[2:5], t[4:-1])
print(t[:2], t[3:])

## 2D Array
t = np.array([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.], [10., 11., 12.]])
pp.pprint(t)
print(t.ndim) # rank
print(t.shape) # shape

## Shape, Rank, Axis
t = tf.constant([1,2,3,4])
pp.pprint(tf.shape(t).eval())

t = tf.constant([[1,2],
                [3,4]])
pp.pprint(tf.shape(t).eval())

t = tf.constant([[
                 [[1,2,3,4],[5,6,7,8], [9,10,11,12]],
                 [[13,14,15,16], [17,18,19,20], [21,22,23,24]]
                ]])
pp.pprint(tf.shape(t).eval())


## Matmul vs Multiply
matrix1 = tf.constant([[1.,2.], [3.,4.]])
matrix2 = tf.constant([[1.],[2.]])
print("Metrix 1 shape", matrix1.shape)
print("Metrix 2 shape", matrix2.shape)
print(tf.matmul(matrix1, matrix2).eval())

print((matrix1*matrix2).eval())

## Broadcasting
matrix1 = tf.constant([[3., 3.]])
matrix2 = tf.constant([[2.],[2.]])
print((matrix1+matrix2).eval())

matrix1 = tf.constant([[[3.,3.]]])
matrix2 = tf.constant([[2.,2.]])
print((matrix1+matrix2).eval())

## Reduce_mean 
print(tf.reduce_mean([1, 2], axis=0).eval())

x = [[1., 2.],
     [3., 4.]]
print(tf.reduce_mean(x).eval())
print(tf.reduce_mean(x, axis=0).eval())
print(tf.reduce_mean(x, axis=1).eval())
print(tf.reduce_mean(x, axis=-1).eval())

## Reduce_sum
x = [[1., 2.],
     [3., 4.]]
print(tf.reduce_sum(x).eval())
print(tf.reduce_sum(x, axis=0).eval())
print(tf.reduce_sum(x, axis=1).eval())
print(tf.reduce_mean(tf.reduce_sum(x, axis=-1)).eval())

## Random
# 평균이 0이고 표준편차가 2인 정규 분포를 따르는 난수로 이루어진 3x3x3 텐서
normal = tf.random_normal([3,3,3], mean=0.0, stddev=2.0)
print(normal.eval())
# 0과 10 사이 난수로 균등하게 이루어진 3x3x3 텐서
uniform = tf.random_uniform([3,3,3], minval=0, maxval=10)
print(uniform.eval())

## ArgMax with axis
x = [[0, 1, 2],
     [2, 1, 0]]
print(tf.argmax(x, axis=0).eval())
print(tf.argmax(x, axis=1).eval())
print(tf.argmax(x, axis=-1).eval())

## Reshape
t = np.array([[[0, 1, 2],
               [3, 4, 5]],
              [[6, 7, 8],
               [9, 10, 11]]])
print(t.shape)
print(tf.reshape(t, shape=[-1, 3]).eval())
print(tf.reshape(t, shape=[-1, 1, 3]).eval())
print(tf.reshape(t, shape=[-1, 2, 3]).eval())

## Reshape(squeeze, expand)
print(tf.squeeze([[0], [1], [2]]).eval())
print(tf.expand_dims([0, 1, 2], 0).eval())
print(tf.expand_dims([0, 1, 2], 1).eval())

## one hot
print(tf.one_hot([[0], [1], [2], [0]], depth=3).eval())
t = tf.one_hot([[0], [1], [2], [0]], depth=3)
print(tf.reshape(t, shape=[-1, 3]).eval())

## casting
print(tf.cast([1.8, 2.2, 3.3, 4.9], tf.int32).eval())
print(tf.cast([True, False, 1==1, 0==1], tf.int32).eval())

## stack
x = [1, 4]
y = [2, 5]
z = [3, 6]

print(tf.stack([x, y, z]).eval())
print(tf.stack([x, y, z], axis=0).eval())
print(tf.stack([x, y, z], axis=1).eval())

## ones and zeros like
x = [[0, 1, 2],
     [2, 1, 0]]
print(tf.ones_like(x).eval())
print(tf.zeros_like(x).eval())

## zip
for x, y in zip([1,2,3], [4,5,6]):
    print(x,y)

for x,y,z in zip([1,2,3], [4,5,6], [7,8,9]):
    print(x,y,z)

