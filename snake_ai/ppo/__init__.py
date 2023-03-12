from datetime import datetime
from itertools import count
from sys import exit
from time import time

import tensorflow as tf
import toml

from snake_ai.ppo.agent import PPOAgent
from snake_game import SnakeGame


class PPOTrainer:

    def __init__(self) -> None:
        config = toml.load("config.toml")
        self.n_iter: int = config['training']['n_iter']
        self.growing_factor: float = config['training']['growing_factor']
        if self.n_iter == 0:
            self.growing_end_epoch: int = config['training']['growing_end_epoch']
        self.env = SnakeGame()
        self.agent = PPOAgent()

    def train(self) -> None:
        print('training started...')
        ratio_increment = 1.0 / ((self.n_iter * self.growing_factor) if self.n_iter != 0 else self.growing_end_epoch)
        start_time = time()
        summary_writer = tf.summary.create_file_writer(f'logs/PPO-{datetime.now().strftime("%Y%m%d-%H%M%S")}')
        looper = range(self.n_iter) if self.n_iter else count()
        try:
            for i in looper:
                board, direction = self.env.reset(min(1, i * ratio_increment))
                done = False
                total_score = 0
                total_step = 0
                while not done:
                    total_step += 1
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
                print(self.env._snake.__len__(), total_score, self.env._curr_size, total_step)
                if i % 100 == 0:
                    self.agent.save_model()
        except KeyboardInterrupt:
            exit()
        finally:
            print('traing done')
            print(f'it took {(time() - start_time) / 60:.3f} minutes')


if __name__ == '__main__':
    p = PPOTrainer()
    p.train()