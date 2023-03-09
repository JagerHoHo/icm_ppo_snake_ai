from pathlib import Path

import numpy as np
from numpy.typing import NDArray
import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow_probability as tfp
import toml

from snake_ai.networks import ActorCritic
from snake_ai.networks import ICM
from snake_ai.ppo.memory import PPOMemory

tf.keras.mixed_precision.set_global_policy('mixed_float16')
tf.keras.backend.set_image_data_format('channels_first')


class PPOAgent:

    def __init__(self) -> None:
        config = toml.load("config.toml")
        self.game_size = config["snake_game"]["size"]
        self.discount_factor = config["ppo"]["discount_factor"]
        self.gae_factor = config["ppo"]["gae_factor"]
        self.clip_factor = config["ppo"]["clip_factor"]
        self.n_epochs = config["ppo"]["n_epochs"]
        self.batch_size = config["ppo"]["batch_size"]
        self.buffer_size = config["ppo"]["buffer_size"]
        self.value_loss_coef = config["ppo"]["value_loss_coef"]
        self.entropy_loss_coef = config["ppo"]["entropy_loss_coef"]
        self.icm_forward_loss_coef = config["icm"]["forward_loss_coef"]
        self.icm_pg_loss_coef = config["icm"]["pg_loss_coef"]
        self.icm_eta = config["icm"]["eta"]
        self.plots_path = Path('plots/model')
        self.plots_path.mkdir(parents=True, exist_ok=True)
        self.model: ActorCritic = ActorCritic(4, self.game_size)
        self.icm: ICM = ICM(4, self.game_size)
        self.reply_buffer = PPOMemory(self.batch_size, self.buffer_size, self.discount_factor, self.gae_factor,
                                      self.game_size + 2)
        self.model.compile(optimizer=tf.keras.mixed_precision.LossScaleOptimizer(tfa.optimizers.AdaBelief(epsilon=1e-12)))
        self.icm.compile(optimizer=tf.keras.mixed_precision.LossScaleOptimizer(tfa.optimizers.AdaBelief()))

    def save_model(self) -> None:
        self.model.save('models/ppo')

    def load_model(self) -> None:
        self.model = tf.keras.models.load_model('models/ppo')

    def set_last_last_board(self, board: NDArray[np.float32]) -> None:
        self.reply_buffer.set_last_last_board(board)

    def store_transition(self, board: NDArray[np.float32], direction: NDArray[np.float32], action: float, prob: float, val: float,
                         reward: float, done: bool) -> None:
        self.reply_buffer.append(board, direction, action, prob, val, reward, done)

    def learn(self) -> None:
        for _ in range(self.n_epochs):
            actions, boards, next_boards = self.reply_buffer.icm_ingredients
            with tf.GradientTape(persistent=True) as tape:
                next_features, next_features_predictions, invs = self.icm((actions, boards, next_boards))
                icm_reward = tf.math.squared_difference(next_features, next_features_predictions)
                icm_reward = tf.math.reduce_mean(icm_reward, axis=1)
                icm_reward *= self.icm_pg_loss_coef * self.icm_eta / 2
                inv_loss = tf.keras.losses.SparseCategoricalCrossentropy()(actions, invs) * (1 - self.icm_forward_loss_coef)
                icm_forward_loss = tf.keras.losses.Huber()(next_features, next_features_predictions) * self.icm_forward_loss_coef
                icm_loss = inv_loss + icm_forward_loss
                icm_loss = self.icm.optimizer.get_scaled_loss(icm_loss)
            icm_params = self.icm.trainable_variables
            icm_grads = tape.gradient(icm_loss, icm_params)
            icm_grads = self.icm.optimizer.get_unscaled_gradients(icm_grads)
            self.icm.optimizer.apply_gradients(zip(icm_grads, icm_params))
            icm_reward = icm_reward.numpy()
            for boards, directions, old_values, old_probs, actions, advantages in self.reply_buffer.sample_batches(icm_reward):
                with tf.GradientTape(persistent=True) as tape:
                    boards = tf.convert_to_tensor(boards)
                    directions = tf.convert_to_tensor(directions)
                    old_probs = tf.convert_to_tensor(old_probs)
                    new_probs, new_values = self.model((boards, directions))
                    distribution = tfp.distributions.Categorical(new_probs)
                    new_probs = distribution.log_prob(actions)
                    entropy_loss = tf.reduce_mean(distribution.entropy()) * self.entropy_loss_coef
                    ratio = tf.math.exp(new_probs - old_probs)
                    weighted_probs = advantages * ratio
                    clipped_probs = tf.clip_by_value(ratio, 1 - self.clip_factor, 1 + self.clip_factor) * advantages
                    clipped_loss = tf.math.minimum(clipped_probs, weighted_probs)
                    clipped_loss = tf.math.reduce_mean(clipped_loss)
                    value_loss = tf.keras.losses.Huber()(advantages + old_values, new_values) * self.value_loss_coef
                    total_loss = -1 * (clipped_loss - value_loss + entropy_loss)
                    total_loss = self.model.optimizer.get_scaled_loss(total_loss)
                params = self.model.trainable_variables
                grads = tape.gradient(total_loss, params)
                grads = self.model.optimizer.get_unscaled_gradients(grads)
                self.model.optimizer.apply_gradients(zip(grads, params))
        self.reply_buffer.clear()

    def choose_action(self, board: NDArray[np.float32], direction: NDArray[np.float32]) -> tuple[int, float, float]:
        board = board.reshape((1,) + board.shape)
        direction = direction.reshape(1, 4)
        probs, value = self.model((board, direction))
        distribution = tfp.distributions.Categorical(probs)
        action = distribution.sample().numpy().item()
        value = value.numpy().item()
        log_prob = distribution.log_prob(action).numpy().item()
        return action, log_prob, value