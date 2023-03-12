import tensorflow as tf


class ActorCritic(tf.keras.Model):

    def __init__(self, n_actions: int, max_size: int) -> None:
        super().__init__()
        self.max_size = max_size + 2
        self.conv1 = tf.keras.layers.Conv2D(32, 3, input_shape=(5, max_size, max_size))
        self.conv2 = tf.keras.layers.Conv2D(32, 3, 2)
        self.conv3 = tf.keras.layers.Conv2D(32, 3)
        self.conv4 = tf.keras.layers.Conv2D(32, 3, 2)
        self.dense1 = tf.keras.layers.Dense(256)
        self.dense2 = tf.keras.layers.Dense(128)
        self.policy = tf.keras.layers.Dense(n_actions)
        self.value = tf.keras.layers.Dense(1)

    def call(self, state: tuple) -> tuple[tf.Tensor, tf.Tensor]:
        board, direction = state

        x = self.conv1(board)
        x = tf.nn.relu6(x)
        x = self.conv2(x)
        x = tf.nn.relu6(x)
        x = self.conv3(x)
        x = tf.nn.relu6(x)
        x = self.conv4(x)
        x = tf.nn.relu6(x)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Concatenate()([x, direction])
        x = self.dense1(x)
        x = tf.nn.relu6(x)
        x = self.dense2(x)
        x = tf.nn.relu6(x)
        policy = self.policy(x)
        policy = tf.keras.layers.Activation('softmax', dtype='float32', name='policy')(policy)
        value = self.value(x)
        value = tf.keras.layers.Activation('linear', dtype='float32', name='value')(value)
        return policy, value

    def plot(self):
        board = tf.keras.Input(shape=(self.max_size, self.max_size, 5), name='board')
        direction = tf.keras.Input(shape=(4), name='direction')
        model = tf.keras.Model(inputs=[board, direction], outputs=self.call((board, direction)))
        return tf.keras.utils.plot_model(model, r'plots/model/ActorCritic.png')
