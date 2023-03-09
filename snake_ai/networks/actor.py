import tensorflow as tf
import tensorflow_addons as tfa


class Actor(tf.keras.Model):

    def __init__(self, n_actions: int, size: int) -> None:
        super().__init__()
        self.size = size + 2
        self.conv1 = tf.keras.layers.Conv2D(32, 3, input_shape=(size, size, 1))
        self.conv2 = tf.keras.layers.Conv2D(32, 3, 2)
        self.conv3 = tf.keras.layers.Conv2D(32, 3)
        self.conv4 = tf.keras.layers.Conv2D(32, 3, 2)
        self.dense1 = tf.keras.layers.Dense(256, kernel_regularizer='l2')
        self.dense2 = tf.keras.layers.Dense(128, kernel_regularizer='l2')
        self.dense3 = tf.keras.layers.Dense(n_actions)

    def call(self, state: tuple):
        board, direction = state

        x = self.conv1(board)
        x = tfa.activations.rrelu(x)  # type: ignore
        x = self.conv2(x)
        x = tfa.activations.rrelu(x)  # type: ignore
        x = self.conv3(x)
        x = tfa.activations.rrelu(x)  # type: ignore
        x = self.conv4(x)
        x = tfa.activations.rrelu(x)  # type: ignore
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Concatenate()([x, direction])
        x = self.dense1(x)
        x = tfa.activations.rrelu(x)  # type: ignore
        x = self.dense2(x)
        x = tfa.activations.rrelu(x)  # type: ignore
        x = self.dense3(x)
        x = tf.keras.layers.Activation('softmax', dtype='float32')(x)
        return x

    def plot(self):
        board = tf.keras.Input(shape=(self.size, self.size, 1), name='board')
        direction = tf.keras.Input(shape=(4), name='direction')
        model = tf.keras.Model(inputs=[board, direction], outputs=self.call((board, direction)))
        return tf.keras.utils.plot_model(model, r'plots/model/Actor.png')