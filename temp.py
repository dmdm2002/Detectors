from tensorflow.keras.layers import Activation, Conv2D, GlobalAveragePooling2D, Dense, Reshape, multiply, Multiply
import tensorflow.keras.backend as K
import tensorflow as tf


class Attention(object):
    def __init__(self, feature_map):
        super(Attention, self).__init__()
        self.feature_map = feature_map

        self.gamma_initializer = tf.zeros_initializer()
        self.gamma_regularizer = None
        self.gamma_constraint = None

        self.gamma = tf.keras.layers.Layer.add_weight(shape=(1,), initializer=self.gamma_initializer, name='gamma',
                                     regularizer=self.gamma_regularizer, constraint=self.gamma_constraint)

    def PAM(self):
        input_shape = self.feature_map.get_shape().as_list()
        _, h, w, filters = input_shape

        b = Conv2D(filters // 8, 1, use_bias=False, kernel_initializer='he_normal')(self.feature_map)
        c = Conv2D(filters // 8, 1, use_bias=False, kernel_initializer='he_normal')(self.feature_map)
        d = Conv2D(filters, 1, use_bias=False, kernel_initializer='he_normal')(self.feature_map)

        vec_b = K.reshape(b, (-1, h * w, filters // 8))
        vec_cT = tf.transpose(K.reshape(c, (-1, h * w, filters // 8)), (0, 2, 1))
        bcT = K.batch_dot(vec_b, vec_cT)

        softmax_bcT = Activation('softmax')(bcT)

        vec_d = K.reshape(d, (-1, h * w, filters))
        bcTd = K.batch_dot(softmax_bcT, vec_d)
        bcTd = K.reshape(bcTd, (-1, h, w, filters))

        out = self.gamma * bcTd + self.feature_map

        return out

    def CAM(self):
        input_shape = self.feature_map.get_shape().as_list()

        _, h, w, filters = input_shape

        vec_a = K.reshape(self.feature_map, (-1, h * w, filters))
        vec_aT = tf.transpose(vec_a, (0, 2, 1))
        aTa = K.batch_dot(vec_aT, vec_a)
        softmax_aTa = Activation('softmax')(aTa)
        aaTa = K.batch_dot(vec_a, softmax_aTa)
        aaTa = K.reshape(aaTa, (-1, h, w, filters))

        out = self.gamma * aaTa + self.feature_map

        return out