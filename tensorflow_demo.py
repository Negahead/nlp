import tensorflow as tf


def foo():
    """
    A TensorFlow program is typically split into two parts: the first part builds a computation
    graph, and the second part runs it. the construction phase typically builds a computation
    graph representing the ML model and the computations required to train it.
    :return:
    """
    x = tf.Variable(3, name='X')
    y = tf.Variable(4, name='y')
    f = x*x*y + y + 2
    sess = tf.Session()
    sess.run(x.initializer)
    sess.run(y.initializer)
    result = sess.run(f)
    print(result)
    sess.close()


def lifecycle_of_a_node_value():
    # When you evaluate a node, TensorFlow automatically determines the set of nodes that it depends
    # on and it evaluates these nodes first.
    w = tf.constant(3)
    x = w + 3
    y = x + 5
    # this will not reuse the result of the previous evaluation of w and x,
    # In short, the preceding code evaluates w and x twice
    z = x * 3
    # all node values are dropped between graph runs, except variable values.
    with tf.Session() as sess:
        y_val, z_val = sess.run([y, z])
        print(y_val)
        print(z_val)


if __name__ == '__main__':
    lifecycle_of_a_node_value()