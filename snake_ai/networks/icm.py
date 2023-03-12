from numpy import float32
from numpy.typing import NDArray
import tensorflow as tf
import tensorflow_addons as tfa


class ICM(tf.keras.Model):

    def __init__(self, n_actions: int, max_size: int) -> None:
        super().__init__()
        self.n_actions = n_actions
        self.max_size = max_size + 2
        self.conv1 = tf.keras.layers.Conv2D(32, 3, 2, input_shape=(4, max_size, max_size), padding='same')
        self.conv2 = tf.keras.layers.Conv2D(32, 3, 2, padding='same')
        self.conv3 = tf.keras.layers.Conv2D(32, 3, 2, padding='same')
        self.features = tf.keras.layers.Conv2D(32, 3, 2, padding='same', name='feature')
        self.inverse1 = tf.keras.layers.Dense(256)
        self.inverse2 = tf.keras.layers.Dense(n_actions)
        self.next_features_prediction1 = tf.keras.layers.Dense(256)
        self.next_features_prediction2 = tf.keras.layers.Dense(self._features_shape * self._features_shape * 32)

    def feature_extract(self, state):
        features = self.conv1(state)
        features = tf.nn.relu6(features)
        features = self.conv2(features)
        features = tf.nn.relu6(features)
        features = self.conv3(features)
        features = tf.nn.relu6(features)
        features = self.features(features)
        features = tf.keras.layers.Flatten()(features)
        return features

    def call(self, inputs: tuple[NDArray[float32], NDArray[float32], NDArray[float32]]):
        action, board, next_board = inputs
        features = self.feature_extract(board)
        next_features = self.feature_extract(next_board)
        next_features = tf.keras.layers.Activation('linear', dtype='float32')(next_features)
        inv = tf.keras.layers.Concatenate()([features, next_features])
        inv = self.inverse1(inv)
        inv = self.inverse2(inv)
        inv = tf.keras.layers.Activation('softmax', dtype='float32')(inv)
        next_features_prediction = tf.keras.layers.Concatenate()([features, action])
        next_features_prediction = self.next_features_prediction1(next_features_prediction)
        next_features_prediction = self.next_features_prediction2(next_features_prediction)
        next_features_prediction = tf.keras.layers.Activation('linear', dtype='float32')(next_features_prediction)
        return next_features, next_features_prediction, inv

    @property
    def _features_shape(self) -> int:
        shape = self.max_size
        for _ in range(4):
            shape = (shape - 1) // 2 + 1
        return shape

    def plot(self):
        action = tf.keras.Input(shape=(self.n_actions), name='action')
        state = tf.keras.Input(shape=(self.max_size, self.max_size, 4), name='state')
        next_state = tf.keras.Input(shape=(self.max_size, self.max_size, 4), name='next_state')
        model = tf.keras.Model(inputs=[action, state, next_state], outputs=self.call((action, state, next_state)))
        return tf.keras.utils.plot_model(model, r'plots/model/ICM.png')
