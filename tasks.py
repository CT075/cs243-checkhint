import ast
from dataclasses import dataclass

from abc import ABC, abstractmethod


class _Task(ABC):
    @abstractmethod
    def is_empty(self) -> bool:
        ...

    def __bool__(self):
        return not self.is_empty()

    @abstractmethod
    def rebuild(self) -> list[ast.stmt]:
        ...


@dataclass
class Basic(_Task):
    body: list[ast.stmt]

    def is_empty(self) -> bool:
        return not self.body

    def rebuild(self) -> list[ast.stmt]:
        return self.body


@dataclass
class While(_Task):
    test: ast.expr
    body: list["Task"]
    num_iterations: int

    def is_empty(self):
        return not self.body

    def rebuild(self) -> list[ast.stmt]:
        return [
            ast.While(self.test, sum(subtask.rebuild() for subtask in self.body), [])
        ]


@dataclass
class For(_Task):
    target: ast.expr
    source: ast.expr
    body: list["Task"]
    num_iterations: int

    def is_empty(self):
        return not self.body

    def rebuild(self) -> list[ast.stmt]:
        return [
            ast.For(
                self.target,
                self.source,
                sum(subtask.rebuild() for subtask in self.body),
                [],
                None,
            )
        ]


@dataclass
class If(_Task):
    test: ast.expr
    body: list["Task"]
    orelse: list["Task"]

    def is_empty(self):
        return not (self.body or self.orelse)

    def rebuild(self) -> list[ast.stmt]:
        return [
            ast.If(
                self.test,
                sum(subtask.rebuild() for subtask in self.body),
                sum(subtask.rebuild() for subtask in self.orelse),
            )
        ]


@dataclass
class With(_Task):
    items: list[ast.withitem]
    body: list["Task"]

    def is_empty(self):
        return not self.body

    def rebuild(self) -> list[ast.stmt]:
        return [ast.With(self.items, sum(subtask.rebuild() for subtask in self.body), None)]


Task = Basic | While | For | If | With
