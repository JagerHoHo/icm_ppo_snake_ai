import tensorflow as tf


class NoisyDense(tf.keras.layers.Layer):

    def __init__(self, units, sigma=0.5):
        super(NoisyDense, self).__init__()
        self.units = units
        self.sigma = sigma

    def build(self, input_shape):
        # initialize weight parameters
        self.mu_w = self.add_weight(shape=(input_shape[-1], self.units), initializer='random_uniform', trainable=True)
        self.sigma_w = self.add_weight(shape=(input_shape[-1], self.units),
                                       initializer=tf.keras.initializers.Constant(self.sigma / tf.math.sqrt(input_shape[-1])),
                                       trainable=True)
        # initialize bias parameters
        self.mu_b = self.add_weight(shape=(self.units,), initializer='zeros', trainable=True)
        self.sigma_b = self.add_weight(shape=(self.units,),
                                       initializer=tf.keras.initializers.Constant(self.sigma / tf.math.sqrt(self.units)),
                                       trainable=True)

    def call(self, inputs, training=True):
        if training:  # check this condition
            # sample noise for weights and biases
            epsilon_w = tf.random.normal((inputs.shape[-1], self.units))
            epsilon_b = tf.random.normal((self.units,))

            # compute noisy weights and biases
            w = self.mu_w + tf.math.multiply(self.sigma_w, epsilon_w)
            b = self.mu_b + tf.math.multiply(self.sigma_b, epsilon_b)
        else:
            # use mean weights and biases without noise
            w = self.mu_w
            b = self.mu_b

        return tf.matmul(inputs, w) + b