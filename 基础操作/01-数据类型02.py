import tensorflow as tf
import numpy as np
with tf.device('cpu'):
    a=tf.constant([1])
with tf.device('gpu'):
    b=tf.constant(4)

print(a.device)
print(b.device)
'''
aa=a.gpu()
bb=b.cpu()
print(aa.device())
print(bb.device())
'''
b.numpy()
print(b.numpy())

print(a.ndim)

print(tf.rank(a))

print(tf.rank(tf.ones([3,4,2])))

c=tf.Variable(9.999,name='vasr')
print(c.name)

d=tf.constant([1.])
e=tf.constant([True,False])
f=tf.constant('hello wold.')
g=np.arange(4)

print(isinstance(d,tf.Tensor))
print(tf.is_tensor(e))
print(tf.is_tensor(g))

print(d.dtype)
print(e.dtype)
print(g.dtype)

if(d.dtype==tf.float32):
    print('True')
else:
    print('False')

if(e.dtype==tf.string):
    print('True')
else:
    print('False')

aaa=np.arange(5)
print(aaa.dtype)

aaaa=tf.convert_to_tensor(aaa)
print(aaaa)
aaaa=tf.convert_to_tensor(aaa,dtype=tf.int64)
print(aaaa)

print(tf.cast(aaaa,dtype=tf.float32))

aaaaa=tf.cast(aaaa,dtype=tf.double)
print(aaaaa)

h=tf.constant([0,1])
print(tf.cast(h,dtype=tf.bool))
hh=tf.cast(h,dtype=tf.bool)
print(tf.cast(hh,tf.int32))

y=tf.range(5)
print(y)

x=tf.Variable(y)
print(x.dtype,x.name)
print(x.trainable)

print(isinstance(x,tf.Tensor))
print(isinstance(y,tf.Variable))

k=tf.ones([])   #标量
print(k)
