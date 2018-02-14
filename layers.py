import tensorflow as tf


class Dense():
    """ Fully-connected layer """
    def __init__(self, scope="dense_layer", idx=0, size=None, dropout=1., nonlinearity=tf.identity):
        assert size, "You must specify a layer size of %s" % scope
        self.scope = scope
        self.size = size
        self.dropout = dropout
        self.nonlinearity = nonlinearity
        self.name = scope + "_" + repr(idx)
        print(" Creating dense layer %s: size = %d" % (self.name, size))

    def __call__(self, x):
        """ Apply dense layer to the input tensor `x` """
        with tf.name_scope(self.scope):
            with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
                self.w = tf.get_variable("weights", shape=[x.get_shape()[1].value, self.size], initializer=tf.contrib.layers.xavier_initializer())
                self.b = tf.get_variable("biases", [self.size], initializer=tf.constant_initializer(0.0))
            layer = self.nonlinearity(tf.matmul(x, self.w) + self.b)
            #return tf.nn.dropout(self.nonlinearity(tf.matmul(x, self.w) + self.b), self.dropout)
            #return self.nonlinearity(tf.matmul(x, self.w) + self.b)
            return tf.nn.dropout(layer, self.dropout)
