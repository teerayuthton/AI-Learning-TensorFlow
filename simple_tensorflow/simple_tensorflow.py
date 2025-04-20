import tensorflow as tf

# 1D Tensor
d1 = tf.ones((2,))
#[1. 1.]

# 2D Tensor
d2 = tf.ones((2, 2))
#[[1. 1.]
# [1. 1.]]

# 3D Tensor
d3 = tf.ones((2, 2, 2))
#[[[1. 1.]
#  [1. 1.]]
#
# [[1. 1.]
#  [1. 1.]]]

from tensorflow import constant

# Define a 2x3 constant tensor
a2x3 = constant(3, shape=[2, 3])
# [[3 3 3]
#  [3 3 3]]

# Define a 2x2 constant tensor
b2x2 = constant([1, 2, 3, 4], shape=[2, 2])
#[[1 2]
# [3 4]]

import tensorflow as tf

# Define variables
af32 = tf.Variable([1, 2, 3, 4, 5, 6], dtype=tf.float32)    #af32:  [1. 2. 3. 4. 5. 6.]
ai16 = tf.Variable([1, 2, 3, 4, 5, 6], dtype=tf.int16)      #ai16:  [1 2 3 4 5 6]

# Define a constant
b = tf.constant(2, dtype=tf.float32) #b:  2.0

# Compute their product
c0 = tf.multiply(af32, b) #c0:  [ 2.  4.  6.  8. 10. 12.]
c1 = af32 * b             #c1:  [ 2.  4.  6.  8. 10. 12.]

#Summing over tensor dimentions

from tensorflow import ones, reduce_sum

A = ones([2, 3, 4])

#Sum over all dimentions
B_reduce = reduce_sum(A) # 24
#24.0

#Sum over dimentions 0, 1, and 2
B0_reduce = reduce_sum(A, 0)
#[2. 2. 2. 2.]
# [2. 2. 2. 2.]
# [2. 2. 2. 2.]]

B1_reduce = reduce_sum(A, 1)
#[[3. 3. 3. 3.]
# [3. 3. 3. 3.]]

B2_reduce = reduce_sum(A, 2)
#[[4. 4. 4.]
# [4. 4. 4.]]