import numpy as np
import tensorflow as tf

a = np.array([1,2,3], dtype="int32")
print(a)

b = np.array([[9.0, 8.0,7.0],[6.0,5.0,4.0]])
print(b)

print(a.ndim)

print(b.shape)
print(b.ndim)

print(a.dtype)
print(b.dtype)

##======================

a = np.array([[1,2,3,4,5,6,7],
              [8,9,10,11,12,13,14]])
print(a)

print(a[1,5])

print(a[:,2])
print(a[0,1:-1:2])

##======================
## Accessing between numpy and tensor
print("Accessing between numpy and tensor")
b = np.array([[[1,2],[3,4]],[[5,6],[7,8]]])
print(b)
print(b[0,1,1])
print(b[0,:,:])
print(b[:,0,:])
print(b[:,:,0])

b = tf.constant([[[1,2],[3,4]],[[5,6],[7,8]]])
print(b)
print(b[0,:,:])

## replace
b = np.array([[[1,2],[3,4]],[[5,6],[7,8]]])
b[:,0,:] = [[0,0],[0,0]]
print(b)

before = np.array([[1,2,3],[5,6,7]])
print(before, before.shape)

after = before.reshape((3,2))
print(after)

## dot mul
# Vertically stacking vectors in numpy
v1 = np.array([1,2,3,4])
v2 = np.array([5,6,7,8])
v3 = np.vstack([v1,v2,v1,v2])
print(v3, v3.shape)

# Vertically stacking vectors in tensor
v1 = tf.constant([1,2,3,4])
v2 = tf.constant([5,6,7,8])
v3 = tf.stack([v1,v2,v1,v2], axis=0)
print(v3, v3.shape)

# Horizontal stack in numpy
h1 = np.ones((2,4))
h2 = np.zeros((2,2))
h3 = np.hstack([h1,h2])
print(h3, h3.shape)

# Horizontal stack in tenfor
h1 = tf.ones((2,4))
h2 = tf.zeros((2,2))
h3 = tf.concat([h1,h2], axis=1)
print(h3)

## Initializing
a = np.array([[1,2,3],[4,5,6]])
b = np.full((2,3),4)
c = np.full_like(a,4)
print(a,b,c)

a = tf.constant([[1,2,3],[4,5,6]])
b = tf.fill(a.shape,4)
print(a,b)

## randomly initialize
## 0~1사이의 random한 값을 (4,2) 모양의 행렬로 초기화
a = np.random.uniform(0,1,(4,2))
b = tf.random.uniform((4,2),0,1)

## Simple Calculations
an = np.array([1,2,3,4])
bn = np.cos(an)
print(bn)

at = tf.constant([1,2,3,4], dtype="float64")
bt = tf.cos(at)
print(bt)

print((an + 2))

print((an ** 2))

b = tf.constant([1,0,1,0])
print(an+b)
a = tf.constant([1,2,3,4], dtype="float16")
print(tf.cos(a))

## Aggregation
## |-----------> axis=1
## |  1  2  3
## |  4  5  6
## |
## axis=0
stats = np.array([[1,2,3],[4,5,6]])
print(stats)

## 전체에서 가장 작은 값 추출 (1)
np.min(stats)

## 행 기준 가장 큰값 추출 ([3,6])
np.max(stats, axis=1)

## 열 기준 합산 ([5,7,9])
np.sum(stats, axis=0)

stats = tf.constant([[1,2,3],[4,5,6]])
print(stats.numpy())

print(tf.reduce_min(stats))
print(tf.reduce_max(stats, axis=1))
print(tf.reduce_sum(stats, axis=0))

stats = tf.constant([[[1,2,3],[4,5,6]],[[1,2,3],[4,5,6]]])
## [[1,2,3],[1,2,3]]
print(tf.reduce_min(stats, axis=1))
## [1,2,3]
print(tf.reduce_min(stats, axis=(0,1)))
## [1,1]
print(tf.reduce_min(stats, axis=(1,2)))

a = np.array([[1,2,3,4,5,6,7],[8,9,10,11,12,13,14]])
print(a[:,:,0])