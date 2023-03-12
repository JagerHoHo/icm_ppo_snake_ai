from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from functools import lru_cache
from functools import singledispatchmethod
from itertools import combinations_with_replacement
from typing import Generator

__all__ = ['Point']


@dataclass
class Point:
    x: int
    y: int

    def __iter__(self) -> Generator[int, None, None]:
        yield self.x
        yield self.y

    @singledispatchmethod
    def __gt__(self: Point, other) -> bool:
        raise NotImplementedError(f"Cannot compare Point to a {type(other)}")

    @__gt__.register
    def _(self, other: int) -> bool:
        return self.x > other or self.y > other

    @singledispatchmethod
    def __lt__(self, other) -> bool:
        raise NotImplementedError(f"Cannot compare Point to a {type(other)}")

    @__lt__.register
    def _(self, other: int) -> bool:
        return self.x <= other or self.y <= other

    @singledispatchmethod
    def __ge__(self: Point, other) -> bool:
        raise NotImplementedError(f"Cannot compare Point to a {type(other)}")

    @__ge__.register
    def _(self, other: int) -> bool:
        return self.x >= other or self.y >= other

    @singledispatchmethod
    def __le__(self, other) -> bool:
        raise NotImplementedError(f"Cannot compare Point to a {type(other)}")

    @__le__.register
    def _(self, other: int) -> bool:
        return self.x <= other or self.y <= other
