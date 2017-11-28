import data_generate
import tensorflow as tf
import numpy
data, stim, A, B, C, D =data_generate.CDN_generate(seed=123,n_area=6,SNR=0.001,A_u=True,B_u=False,C_u=False,D_u=True, which_data=3) 

n_sim = 100000

n_channels = 1
seq_long = data.shape[1]
n_nodes = data.shape[0]

A_t = tf.placeholder(shape=[None,A.shape[0]*A.shape[1]],dtype=tf.float32)

x = tf.placeholder(shape=[None,n_nodes,seq_long,n_channels],dtype=tf.float32)


x_list = tf.split(x,num_or_size_splits = n_nodes, axis =1)



def mynet(inpt):
    W1 = tf.Variable(tf.random_uniform([2,8,1,128],dtype=tf.float32))
    b1 = tf.Variable(tf.truncated_normal([128], stddev=0.05))
    W2 = tf.Variable(tf.random_uniform([3,128,128],dtype=tf.float32))
    b2 = tf.Variable(tf.truncated_normal([128], stddev=0.05))
    W3 = tf.Variable(tf.random_uniform([3,128,128],dtype=tf.float32))
    b3 = tf.Variable(tf.truncated_normal([128], stddev=0.05))
    W4 = tf.Variable(tf.random_uniform([1408,1],dtype=tf.float32))
    b4 = tf.Variable(tf.truncated_normal([1], stddev=0.05))
    
    conv1 = tf.nn.tanh(tf.nn.conv2d(inpt, W1, strides=[1,2,1,1], padding='SAME')+b1)
    maxpool1 = tf.nn.max_pool(conv1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
    maxpool1 = tf.squeeze(maxpool1,axis=1)
    conv2 = tf.nn.tanh(tf.nn.conv1d(maxpool1, W2, stride=1, padding='SAME')+b2)
    maxpool2 = tf.layers.max_pooling1d(inputs=conv2, pool_size=3, strides=3, padding='SAME')
    conv3 = tf.nn.tanh(tf.nn.conv1d(maxpool2, W3, stride=1, padding='SAME')+b3)
    maxpool3 = tf.layers.max_pooling1d(inputs=conv3, pool_size=3, strides=3, padding='SAME')
    flatten = tf.contrib.layers.flatten(maxpool3)
    fc = tf.nn.tanh(tf.matmul(flatten, W4)+b4)
    return fc

x_concat_list = []

with tf.variable_scope("cnn1d") as scope:
  for i,t_i in enumerate(x_list):
    for j,t_j in enumerate(x_list):
      exec('x_concat_%d_%d = tf.concat([t_i,t_j],1)' %(i,j) )
      exec('x_%d_%d = mynet(x_concat_%d_%d)' %(i,j,i,j))
      scope.reuse_variables()
      exec('x_concat_list.append(x_%d_%d)' %(i,j))

x_=tf.stack(x_concat_list,axis=1)
x_pred = tf.reshape(x_,[-1,36])

W5 = tf.Variable(tf.random_uniform([36,36],dtype=tf.float32))
x_pred = tf.matmul(x_pred, W5)


loss = tf.norm((A_t - x_pred),ord=2,axis =1,keep_dims=True)
optimizer = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(loss)

saver=tf.train.Saver()

with tf.Session() as sess:
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    sess.run(init_op)
    #saver.restore(sess,"/gpfs/scratch/jke/deep/ode/ode.ckpt")
    for batch_index in range(n_sim):
        data, stim, A, B, C, D =data_generate.CDN_generate(seed=batch_index,n_area=6,SNR=0.0001,A_u=True,B_u=False,C_u=False,D_u=True,which_data=1) 
        data = data.reshape(1,6,184,1)
        A_norm = numpy.linalg.norm(A,ord=2)
        A = A.reshape(1,36)
        train_loss, _=sess.run([loss, optimizer],feed_dict={x:data,A_t:A})
        print("loss: %s" % str(train_loss))
        print(float(train_loss)/A_norm)
        if batch_index%200==0: saver.save(sess,"/gpfs/scratch/jke/deep/ode/ode.ckpt")
    sess.close()
