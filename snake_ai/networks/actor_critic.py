import tensorflow as tf
import tensorflow_addons as tfa


class ActorCritic(tf.keras.Model):

    def __init__(self, n_actions: int, size: int) -> None:
        super().__init__()
        self.size = size + 2
        self.conv1 = tf.keras.layers.Conv2D(32, 3, input_shape=(5, size, size))
        self.conv2 = tf.keras.layers.Conv2D(32, 3, 2)
        self.conv3 = tf.keras.layers.Conv2D(32, 3)
        self.conv4 = tf.keras.layers.Conv2D(32, 3, 2)
        self.dense1 = tf.keras.layers.Dense(256, kernel_regularizer='l2')
        self.dense2 = tf.keras.layers.Dense(128, kernel_regularizer='l2')
        self.policy = tf.keras.layers.Dense(n_actions)
        self.value = tf.keras.layers.Dense(1)

    def call(self, state: tuple) -> tuple[tf.Tensor, tf.Tensor]:
        board, direction = state

        x = self.conv1(board)
        x = tfa.activations.rrelu(x)
        x = self.conv2(x)
        x = tfa.activations.rrelu(x)
        x = self.conv3(x)
        x = tfa.activations.rrelu(x)
        x = self.conv4(x)
        x = tfa.activations.rrelu(x)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Concatenate()([x, direction])
        x = self.dense1(x)
        x = tfa.activations.rrelu(x)
        x = self.dense2(x)
        x = tfa.activations.rrelu(x)
        policy = self.policy(x)
        policy = tf.keras.layers.Activation('softmax', dtype='float32', name='policy')(policy)
        value = self.value(x)
        value = tf.keras.layers.Activation('linear', dtype='float32', name='value')(value)
        return policy, value

    def plot(self):
        board = tf.keras.Input(shape=(self.size, self.size, 5), name='board')
        direction = tf.keras.Input(shape=(4), name='direction')
        model = tf.keras.Model(inputs=[board, direction], outputs=self.call((board, direction)))
        return tf.keras.utils.plot_model(model, r'plots/model/ActorCritic.png')
