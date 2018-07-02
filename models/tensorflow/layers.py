import tensorflow as tf

def conv(data, ksize, filters, ssize, padding, use_bias, conv_name=None, bn_name=None, bn=False, act=True):
    if not bn :
        if act : 
            output = tf.layers.conv2d(data, kernel_size=ksize, filters=filters,
                                      strides=(ssize,ssize),
                                      padding=padding,
                                      name=conv_name, 
                                      activation=tf.nn.relu,use_bias=use_bias,
                                      kernel_initializer=tf.contrib.layers.xavier_initializer())
        else : 
            output = tf.layers.conv2d(data, kernel_size=ksize, filters=filters,
                                      strides=(ssize,ssize),
                                      padding=padding,
                                      name=conv_name,use_bias=use_bias,
                                      kernel_initializer=tf.contrib.layers.xavier_initializer())
    else : 
        conv = tf.layers.conv2d(data, kernel_size=ksize, filters=filters,
                                strides=(ssize,ssize),
                                padding=padding,
                                name=conv_name,use_bias=use_bias,
                                kernel_initializer=tf.contrib.layers.xavier_initializer())
        with tf.variable_scope(bn_name) as bn_name:
            output = tf.contrib.layers.batch_norm(conv)
        if act : output = tf.nn.relu(output)
    return output

def deconv(data, ksize, filters, ssize, padding, use_bias, deconv_name=None, bn_name=None, bn=False, act=True):
    if not bn :
        if act : 
            output = tf.layers.conv2d_transpose(data, kernel_size=ksize, filters=filters,
                                                strides=(ssize,ssize),
                                                padding=padding,
                                                name=deconv_name, 
                                                activation=tf.nn.relu,use_bias=use_bias,
                                                kernel_initializer=tf.contrib.layers.xavier_initializer())
        else : 
            output = tf.layers.conv2d_transpose(data, kernel_size=ksize, filters=filters,
                                                strides=(ssize,ssize),
                                                padding=padding,
                                                name=deconv_name,use_bias=use_bias,
                                                kernel_initializer=tf.contrib.layers.xavier_initializer())
    else : 
        deconv = tf.layers.conv2d_transpose(data, kernel_size=ksize, filters=filters,
                                          strides=(ssize,ssize),
                                          padding=padding,
                                          name=deconv_name,use_bias=use_bias,
                                          kernel_initializer=tf.contrib.layers.xavier_initializer())
        with tf.variable_scope(bn_name) as bn_name:
            output = tf.contrib.layers.batch_norm(deconv)
        if act : output = tf.nn.relu(output)
    return output

def max_pooling(data, name=None):
    return tf.nn.max_pool(data, ksize=[1,3,3,1], strides=[1,2,2,1], padding="SAME", name=name)

def dropout(data, name=None, ratio=0.5):
    return tf.nn.dropout(data, ratio, name=name)

def lrn(data, depth_radius, alpha, beta, name):
    return tf.nn.local_response_normalization(data, depth_radius=depth_radius, alpha=alpha, beta=beta, bias=1.0, name=name)

def bn(data, name=None):
    with tf.variable_scope(name) as name:
        batch_norm = tf.contrib.layers.batch_norm(data)
    return batch_norm

def fc(data, num_in, num_out, name=None, relu=True):
    with tf.variable_scope(name) as scope:
        weights = tf.get_variable('weights', shape=[num_in, num_out], trainable=True)
        biases = tf.get_variable('biases', [num_out], trainable=True)
        output = tf.nn.xw_plus_b(data, weights, biases, name=scope.name)
    if relu : 
        return tf.nn.relu(output)
    else: 
        return output

def ZeroPadding2D(data,psize,Type="CONSTANT",name=None):
    data_shape=data.get_shape().as_list()
    if data_shape[0]==None :
        raise ValueError("batch_size must be specified.")
    else:
        paddings = tf.constant([psize, psize])
        for batch_size in range(data_shape[0]):
            padded = tf.pad(data[0,:,:,0],paddings,Type)
            padded = tf.expand_dims(tf.expand_dims(padded,axis=0),axis=3)

            for channels in range(1,data_shape[3]):
                padded_2 = tf.pad(data[0,:,:,channels],paddings,Type)
                padded_2 = tf.expand_dims(tf.expand_dims(padded_2,axis=0),axis=3)
                padded = tf.concat([padded,padded_2],3)

            if batch_size==0 : 
                padded_dataset=padded
            else : 
                padded_dataset=tf.concat([padded_dataset,padded],0)
    return padded_dataset