from datetime import datetime

import tensorflow as tf
import toml

from snake_ai.ppo.agent import PPOAgent
from snake_game import SnakeGame


class PPOTrainer:

    def __init__(self) -> None:
        config = toml.load("config.toml")
        self.n_iter = config["ppo"]["n_iter"]
        self.env = SnakeGame()
        self.agent = PPOAgent()
        self.log_dir = "logs/"

    def train(self) -> None:
        summary_writer = tf.summary.create_file_writer(f'{self.log_dir}train/{datetime.now().strftime("%Y%m%d-%H%M%S")}')
        for i in range(self.n_iter):
            board, direction = self.env.reset()
            done = False
            score = 0
            while not done:
                action, prob, val = self.agent.choose_action(board, direction)
                (next_board, next_direction), reward, done = self.env.step(action)
                score += reward
                self.agent.store_transition(board, direction, action, prob, val, reward, done)
                if self.agent.reply_buffer.is_full:
                    self.agent.set_last_last_board(next_board)
                    self.agent.learn()
                board, direction = next_board, next_direction
            with summary_writer.as_default():
                tf.summary.scalar('total score', score, step=i)
            print(score)


if __name__ == '__main__':
    p = PPOTrainer()
    p.train()