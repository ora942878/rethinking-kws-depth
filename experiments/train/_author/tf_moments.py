"""TensorFlow counterpart of the original BC8 moment objective."""
import tensorflow.compat.v1 as tf

def regularizer(outputs, pairs):
    eps = 1e-05
    moments = [tf.nn.moments(tf.cast(x, tf.float32), axes=list(range(x.shape.rank - 1))) for x in outputs]
    terms = []
    for n, r in pairs:
        m, v = moments[n - 1]
        mr, vr = moments[r - 1]
        terms.append(tf.reduce_mean((tf.square(m - mr) + tf.square(tf.sqrt(v + eps) - tf.sqrt(vr + eps))) / (v + vr + eps)))
    return tf.add_n(terms) / len(terms)

def kwt_loss(outputs):
    with tf.variable_scope('moment_regularization', reuse=tf.AUTO_REUSE):
        weight = tf.get_variable('weight', shape=(), initializer=tf.zeros_initializer(), trainable=False)
    value = regularizer(outputs, [(i, 1) for i in range(2, 13)])
    tf.add_to_collection('moment_raw_loss', value)
    return weight * value

def tc_loss():
    outputs = tf.get_collection('moment_block_outputs')
    if not outputs:
        return tf.constant(0.0)
    step = tf.train.get_or_create_global_step()
    weight = 0.1 * tf.clip_by_value((tf.cast(step, tf.float32) / (36923 / 100) - 5) / 15, 0.0, 1.0)
    value = regularizer(outputs, [(2, 1), (4, 3), (6, 5)])
    tf.add_to_collection('moment_raw_loss', value)
    return weight * value
