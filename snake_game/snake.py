from __future__ import annotations

from collections import deque
from copy import deepcopy
from dataclasses import dataclass
from dataclasses import field
from enum import IntEnum
from typing import Generator

import numpy as np
from numpy.typing import NDArray

from snake_game.point import Point


class Direction(IntEnum):
    UP = 1
    DOWN = -1
    LEFT = 2
    RIGHT = -2
    CENTER = 0

    @property
    def reverse(self) -> Direction:
        return Direction(self.value * -1)


@dataclass
class Snake:
    head: Point
    starve_endurance: int
    body: deque[Point] = field(init=False, default_factory=deque)
    direction: Direction = field(init=False, default=Direction.CENTER)
    _starve_count: int = field(init=False, default=0)

    def __len__(self) -> int:
        return len(self.body) + 1

    def __iter__(self) -> Generator[Point, None, None]:
        yield self.head
        yield from self.body

    def __contains__(self, point: Point) -> bool:
        return any(node == point for node in self)

    def turn(self, dir: int) -> None:
        dir_mapping = [Direction.UP, Direction.DOWN, Direction.LEFT, Direction.RIGHT]
        if (direction := dir_mapping[dir]) != self.direction.reverse:
            self.direction = direction

    def move(self) -> None:
        self.body.appendleft(deepcopy(self.head))
        if self.direction == Direction.UP:
            self.head.y -= 1
        elif self.direction == Direction.DOWN:
            self.head.y += 1
        elif self.direction == Direction.LEFT:
            self.head.x -= 1
        elif self.direction == Direction.RIGHT:
            self.head.x += 1

    def reset_starve_count(self) -> None:
        self._starve_count = 0

    def leave_tail_point(self) -> None:
        self.body.pop()

    @property
    def clash_into_self(self) -> bool:
        return self.head in self.body

    @property
    def direction_one_hot(self) -> NDArray[np.float32]:
        direction = np.zeros(4, dtype=np.float32)
        if self.direction == Direction.CENTER:
            return direction
        dir_mapping = {Direction.UP: 0, Direction.DOWN: 1, Direction.LEFT: 2, Direction.RIGHT: 3}
        direction[dir_mapping[self.direction]] = 1
        return direction

    @property
    def starved(self) -> bool:
        return self._starve_count >= self.starve_endurance