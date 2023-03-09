from datetime import datetime
from itertools import count

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
        looper = range(self.n_iter) if self.n_iter else count()
        for i in looper:
            board, direction = self.env.reset()
            done = False
            total_score = 0
            while not done:
                action, prob, val = self.agent.choose_action(board, direction)
                (next_board, next_direction), reward, done = self.env.step(action)
                total_score += reward
                self.agent.store_transition(board, direction, action, prob, val, reward, done)
                if self.agent.reply_buffer.is_full:
                    self.agent.set_last_last_board(next_board)
                    self.agent.learn()
                board, direction = next_board, next_direction
            with summary_writer.as_default():
                tf.summary.scalar('total score', total_score, step=i)
            print(total_score)
            if i % 100 == 0:
                self.agent.save_model()


if __name__ == '__main__':
    p = PPOTrainer()
    p.train()