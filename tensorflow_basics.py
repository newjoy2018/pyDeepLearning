A = tf.constant(1234)
B = tf.constant([1,2,3,4])
C = tf.constant([[1,2,3],[4,5,6]])

with tf.Session() as sess:
    output = sess.run(C)
    print(output)



E = tf.constant([[2,4]])
F = tf.constant([[3],[5]])
Y = tf.matmul(E,F)

sess = tf.Session()
output = sess.run(Y)
print(output)
sess.close()



import tensorflow as tf

E = tf.constant([[2,4]])
F = tf.constant([[3],[5]])
Y = tf.matmul(E,F)

with tf.Session() as sess:
    with tf.device("/gpu:0"):
        output = sess.run(Y)
        print(output)



def run():
    output = None
    x = tf.placeholder(tf.int32)
    y = tf.placeholder(tf.string)
    print(x)
    with tf.Session() as sess:
        output = sess.run(x,feed_dict={x:666,y:'Hello!'})
        print(output)
    return output
run()


x = tf.constant(3.1416)
y = tf.constant(66)
z = tf.subtract(tf.cast(y,tf.float32),x)  #不同数据类型之间的运算需要先转换

with tf.Session() as sess:
    output = sess.run(z)
    print(output)
