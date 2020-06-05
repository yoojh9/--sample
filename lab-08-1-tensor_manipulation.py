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