from collections import deque
from itertools import combinations_with_replacement
from random import randint
from random import shuffle

import numpy as np
from numpy.typing import NDArray
import toml

from snake_game.point import Point
from snake_game.snake import Snake


class SnakeGame:
    State = tuple[NDArray[np.float32], NDArray[np.float32]]
    __slots__ = ('_MAX_SIZE', '_WINNING_REWARD', '_EATEN_FOOD_REWARD', '_EARLY_DONE_PENALTY', '_LIVING_PENALTY', '_snake',
                 '_food_poses', '_food', '_curr_size')

    def __init__(self) -> None:
        config = toml.load("config.toml")
        self._MAX_SIZE: int = config["snake_game"]["max_size"]
        self._WINNING_REWARD: float = config["snake_game"]["winning_reward"]
        self._EATEN_FOOD_REWARD: float = config["snake_game"]["eaten_food_reward"]
        self._EARLY_DONE_PENALTY: float = config["snake_game"]["early_done_penalty"]
        self._LIVING_PENALTY: float = config["snake_game"]["living_penalty"]
        self._curr_size = self._MAX_SIZE
        self.reset(0.001)

    @property
    def _reach_boder(self) -> bool:
        return self._snake.head > self._curr_size or self._snake.head < 1

    @property
    def _eaten_food(self) -> bool:
        return self._snake.head == self._food

    @property
    def winning(self) -> bool:
        return len(self._snake) == self._curr_size * self._curr_size

    @property
    def early_done(self) -> bool:
        """
        early done if:
        1. the snake hasn't eating for too long
        2. the snake has reached the boder
        3. the snake has clashed into itself
        """
        return self._snake.starved or self._reach_boder or self._snake.clash_into_self

    def _new_food_pos(self) -> Point:
        if (pos := self._food_poses.pop()) not in self._snake:
            return pos
        self._food_poses.appendleft(pos)
        return self._new_food_pos()

    @property
    def _state(self) -> State:
        board = np.zeros((5, self._MAX_SIZE + 2, self._MAX_SIZE + 2), dtype=np.bool8)
        walls = np.where(np.pad(np.zeros((self._curr_size, self._curr_size)), ((1, 1), (1, 1)), constant_values=True))
        board[0, walls[1], walls[0]] = True
        x, y = self._snake.head
        board[1, y, x] = True
        for x, y in self._snake.body:
            board[2, y, x] = True
        x, y = self._food
        board[3, y, x] = True
        board[4] = ~(board[0] | board[1] | board[2] | board[3])
        return (board.astype(np.float32), self._snake.direction_one_hot)

    def reset(self, ratio: float) -> State:
        self._curr_size = max(2, int(self._MAX_SIZE * ratio))
        x, y = randint(1, self._curr_size), randint(1, self._curr_size)
        self._snake = Snake(head=Point(x, y), starve_endurance=self._curr_size * 2)
        self._food_poses = deque(Point(x + 1, y + 1) for x, y in combinations_with_replacement(range(self._curr_size), 2))
        shuffle(self._food_poses)
        self._food = self._new_food_pos()
        return self._state

    def step(self, dir: int) -> tuple[State, float, bool]:
        self._snake.turn(dir)
        self._snake.move()
        if self._eaten_food:
            self._snake.reset_starve_count()
            self._food = self._new_food_pos()
            reward = self._WINNING_REWARD if self.winning else self._EATEN_FOOD_REWARD
        else:
            self._snake.leave_tail_point()
            reward = -1 * self._EARLY_DONE_PENALTY if self.early_done else -1 * self._LIVING_PENALTY
        return self._state, reward, self.early_done or self.winning
