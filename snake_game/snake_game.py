from collections import deque
from dataclasses import dataclass
from dataclasses import field
from itertools import combinations_with_replacement
from random import randint
from random import shuffle

import numpy as np
from numpy.typing import NDArray
import toml

from snake_game.point import Point
from snake_game.snake import Snake

State = tuple[NDArray[np.float32], NDArray[np.float32]]


@dataclass
class SnakeGame:
    SIZE: int = field(init=False)
    _food_poses: deque[Point] = field(init=False)
    _food: Point = field(init=False)
    _snake: Snake = field(init=False)
    _WINNING_REWARD: float = field(init=False)
    _EATEN_FOOD_REWARD: float = field(init=False)
    _EARLY_DONE_PENALTY: float = field(init=False)
    _LIVING_PENALTY: float = field(init=False)

    def __post_init__(self) -> None:
        config = toml.load("config.toml")
        self.SIZE = config["snake_game"]["size"]
        self._WINNING_REWARD = config["snake_game"]["winning_reward"]
        self._EATEN_FOOD_REWARD = config["snake_game"]["eaten_food_reward"]
        self._EARLY_DONE_PENALTY = config["snake_game"]["early_done_penalty"]
        self._LIVING_PENALTY = config["snake_game"]["living_penalty"]
        self.reset()

    @property
    def _reach_boder(self) -> bool:
        return self._snake.head > self.SIZE or self._snake.head < 1

    @property
    def _eaten_food(self) -> bool:
        return self._snake.head == self._food

    def _new_food_pos(self) -> Point:
        if (pos := self._food_poses.pop()) not in self._snake:
            return pos
        self._food_poses.appendleft(pos)
        return self._new_food_pos()

    @property
    def _state(self) -> State:
        board = np.zeros((4, self.SIZE + 2, self.SIZE + 2), dtype=np.float32)
        board[0, ...] = np.pad(np.zeros((self.SIZE, self.SIZE)), ((1, 1), (1, 1)), constant_values=1)
        x, y = self._snake.head
        board[1, y, x] = 1
        for x, y in self._snake.body:
            board[2, y, x] = 1
        x, y = self._food
        board[3, y, x] = 1
        return (board, self._snake.direction_one_hot)

    def reset(self) -> State:
        x, y = randint(1, self.SIZE), randint(1, self.SIZE)
        self._snake = Snake(head=Point(x, y), starve_endurance=self.SIZE * 2)
        self._food_poses = deque(Point(x + 1, y + 1) for x, y in combinations_with_replacement(range(self.SIZE), 2))
        shuffle(self._food_poses)
        self._food = self._new_food_pos()
        return self._state

    def step(self, dir: int) -> tuple[State, float, bool]:
        self._snake.turn(dir)
        self._snake.move()
        winning = len(self._snake) == self.SIZE
        early_done = self._snake.starved or self._reach_boder or self._snake.clash_into_self
        self._snake._starve_count += 1
        if self._eaten_food:
            self._snake.reset_starve_count()
        reward = self._WINNING_REWARD if winning else self._EATEN_FOOD_REWARD if self._eaten_food else -1 * self._EARLY_DONE_PENALTY if early_done else -1 * self._LIVING_PENALTY
        return self._state, reward, early_done or winning
