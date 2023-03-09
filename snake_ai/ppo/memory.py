from dataclasses import dataclass
from dataclasses import field
from typing import Generator

from numpy import bool8
from numpy import float32
from numpy import int32
import numpy as np
from numpy.typing import NDArray

Batch = tuple[NDArray[float32], NDArray[float32], NDArray[float32], NDArray[float32], NDArray[bool8], NDArray[int32]]


@dataclass
class PPOMemory:
    batch_size: int
    buffer_size: int
    discount_factor: float
    gae_factor: float
    board_size: int
    index: int = field(init=False)
    boards: NDArray[float32] = field(init=False)
    directions: NDArray[float32] = field(init=False)
    vals: NDArray[float32] = field(init=False)
    probs: NDArray[float32] = field(init=False)
    actions: NDArray[float32] = field(init=False)
    rewards: NDArray[float32] = field(init=False)
    dones: NDArray[bool8] = field(init=False)
    last_board: NDArray[float32] = field(init=False)

    def __post_init__(self) -> None:
        self.clear()

    @property
    def icm_ingredients(self) -> tuple[NDArray[float32], NDArray[float32], NDArray[float32]]:
        return self.actions, self.boards, np.vstack([self.boards[1:], self.last_board[np.newaxis, :]])

    def _get_gae(self, intrinsic_reward: NDArray[float32] | None = None) -> NDArray[float32]:
        rewards = self.rewards
        if intrinsic_reward is not None:
            rewards += intrinsic_reward
        discounts = [self.discount_factor**i * self.gae_factor**i for i in range(self.buffer_size)]
        return np.array([(discounts[:self.buffer_size - i] *
                          (rewards[i:] + self.discount_factor * np.append(self.vals[i + 1:], 0) *
                           (1 - self.dones[i:]) - self.vals[i:])).sum() for i in range(self.buffer_size - 1)] + [0])

    def sample_batches(self, intrinsic_reward: NDArray[float32] | None = None) -> Generator[Batch, None, None]:
        gae = self._get_gae(intrinsic_reward)
        batches = np.array_split(np.random.permutation(self.buffer_size), self.buffer_size // self.batch_size)
        yield from zip(self.boards[batches], self.directions[batches], self.vals[batches], self.probs[batches],
                       self.actions[batches], gae[batches])

    def append(
        self,
        board: NDArray[float32],
        direction: NDArray[float32],
        action: float,
        probs: float,
        vals: float,
        reward: float,
        done: bool,
    ) -> None:
        self.boards[self.index] = board
        self.directions[self.index] = direction
        self.actions[self.index] = action
        self.probs[self.index] = probs
        self.vals[self.index] = vals
        self.rewards[self.index] = reward
        self.dones[self.index] = done
        self.index += 1

    def clear(self) -> None:
        self.boards = np.empty((self.buffer_size, 5, self.board_size, self.board_size), dtype=float32)
        self.last_board = np.empty((5, self.board_size, self.board_size), dtype=float32)
        self.directions = np.empty((self.buffer_size, 4), dtype=float32)
        self.actions = np.empty((self.buffer_size, 1), dtype=float32)
        self.probs = np.empty(self.buffer_size, dtype=float32)
        self.vals = np.empty(self.buffer_size, dtype=float32)
        self.rewards = np.empty(self.buffer_size, dtype=float32)
        self.dones = np.empty(self.buffer_size, dtype=bool8)
        self.index = 0

    def set_last_last_board(self, last_board) -> None:
        self.last_board = last_board

    @property
    def is_full(self) -> bool:
        return self.index == self.buffer_size