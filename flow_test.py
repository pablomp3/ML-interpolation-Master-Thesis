import tensorflow.compat.v1 as tf
import numpy as np


def group_list(l, group_size):
    """
    :param l:           list
    :param group_size:  size of each group
    :return:            Yields successive group-sized lists from l.
    """
    for i in xrange(0, len(l), group_size):
        yield l[i:i+group_size]

# Python optimisation variables
learning_rate = 0.5
epochs = 10
batch_size = 100

begin_arr = np.load('final_interpolation_dataset_np' + '/encoded_full_z_begin.npy')  # load
print(type(begin_arr), begin_arr.shape)
inter_arr = np.load('final_interpolation_dataset_np' + '/encoded_full_z_interp.npy')  # load
print(type(inter_arr), inter_arr.shape)
end_arr = np.load('final_interpolation_dataset_np' + '/encoded_full_z_end.npy')  # load
print(type(end_arr), end_arr.shape)

limit = int(0.8*len(begin_arr))
begin_arr_train = begin_arr[:limit]
begin_arr_test = begin_arr[limit:]
inter_arr_train = inter_arr[:limit]
inter_arr_test = inter_arr[limit:]
end_arr_train = end_arr[:limit]
end_arr_test = end_arr[limit:]

#print(group_list(begin_arr, batch_size))
#print(type(mnist.train.next_batch(batch_size=batch_size)), mnist.train.next_batch(batch_size=batch_size).shape, mnist.train.next_batch(batch_size=batch_size))

# declare the training data placeholders
# input x - for 28 x 28 pixels = 784
x = tf.placeholder(tf.float32, [None, 784])
# now declare the output data placeholder - 10 digits
#y = tf.placeholder(tf.float32, [None, 10])

#declare the training data placeholders
# input z1 = begining encoded track (1,100)
z1 = tf.placeholder(tf.float32, [None, 1, 100], name='z1')
# input z2 = end encoded track (1,100)
z2 = tf.placeholder(tf.float32, [None, 1, 100], name='z2')
# output z = interpolation encoded track (1,100)
y = tf.placeholder(tf.float32, [None, 1, 100], name='y')

# now declare the weights connecting the input to the hidden layer to output
h1_1 = tf.Variable(tf.random_normal([100, 100], stddev=0.35), name="h1_1")
h1_2 = tf.Variable(tf.random_normal([100, 100], stddev=0.35), name="h1_2")
b1 = tf.Variable(tf.random_normal([100]), name='b1')
h2_1 = tf.Variable(tf.random_normal([100, 100], stddev=0.35), name="h2_1")
h2_2 = tf.Variable(tf.random_normal([100, 100], stddev=0.35), name="h2_2")
b2 = tf.Variable(tf.random_normal([100]), name='b2')

'''
# now declare the weights connecting the input to the hidden layer
W1 = tf.Variable(tf.random_normal([784, 300], stddev=0.03), name='W1')
b1 = tf.Variable(tf.random_normal([300]), name='b1')
# and the weights connecting the hidden layer to the output layer
W2 = tf.Variable(tf.random_normal([300, 10], stddev=0.03), name='W2')
b2 = tf.Variable(tf.random_normal([10]), name='b2')
'''

hidden_sum = tf.add(tf.matmul(z1, h1_1), tf.matmul(z2, h1_2)) # z1*h1 + z2*h2 == [1, 100] + [1, 100] = [1, 100]
hidden_out = tf.add(hidden_sum, b1) # hidden_out = (z1*h1 + z2*h2) + b1 == [1, 100]
hidden_out = tf.nn.relu(hidden_out) # [1, 100]

hidden_out_2 = tf.add(tf.matmul(hidden_out, h2_1), b2)#, tf.matmul(z2, h2_2))  # z1*h1 + z2*h2 == [1, 100] + [1, 100] = [1, 100]
z_pred = tf.nn.softmax(hidden_out_2)
y_ = z_pred

y_clipped = tf.clip_by_value(y_, 1e-10, 0.9999999)
cross_entropy = -tf.reduce_mean(tf.reduce_sum(y * tf.log(y_clipped)
                         + (1 - y) * tf.log(1 - y_clipped), axis=1))

# add an optimiser
optimiser = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cross_entropy)

# finally setup the initialisation operator
init_op = tf.global_variables_initializer()

# define an accuracy assessment operation
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# start the session
with tf.Session() as sess:
   # initialise the variables
   sess.run(init_op)
   #total_batch = int(len(mnist.train.labels) / batch_size)
   total_batch = int(len(begin_arr) / batch_size)
   for epoch in range(100):
        avg_cost = 0
        #for i in range(total_batch):
        #    batch_x, batch_y = mnist.train.next_batch(batch_size=batch_size) #numpy array (100, 784), my_case (100, 1, 100)

        jeje, c = sess.run([optimiser, cross_entropy],
                         #feed_dict={x: batch_x, y: batch_y})
                         feed_dict={z1: begin_arr_train, z2:end_arr_train, y: inter_arr_train})
            #avg_cost += c / total_batch
        avg_cost += c
        print("Epoch:", (epoch + 1), "cost =", "{:.3f}".format(avg_cost))
   #print(sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels}))
   print(sess.run(accuracy, feed_dict={z1: begin_arr_test, z2:end_arr_test, y: inter_arr_test}))
