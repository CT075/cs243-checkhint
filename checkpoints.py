import ast
import math
from functools import cache
from dataclasses import dataclass

from typing import assert_never, Tuple, Sequence, Callable

from tasks import Task, Basic, While, For, If, With
from task_splitter import extract_iteration_count
from liveness import live_in_task, initial_live_out


@dataclass
class Program:
    name: str
    decls: list[ast.FunctionDef]
    body: Sequence[ast.stmt]
    origin: Callable

    def __call__(self, *args, **kwargs):
        return self.origin(*args, **kwargs)

    def expand(self) -> str:
        result = ast.Module(
            self.decls
            + [
                ast.FunctionDef(
                    self.name,
                    [],  # .
                    list(self.body),
                    [],
                    None,  # .
                )
            ],
            type_ignores=[],
        )
        ast.fix_missing_locations(result)
        return ast.unparse(result)


def configure_tasks_by_checkpoint(
    name: str,
    origin: Callable,
    tasks: list[Task],
    allowance: float,
    checkpoint_cost: int,
    failure_prob: float,
) -> Program:
    time_blocks = compute_times(tasks)
    times = [t.t for t in time_blocks]
    num_checkpoints = max_checkpoints(times, allowance, checkpoint_cost, failure_prob)
    block_ends = select_block_ends(times, num_checkpoints, failure_prob)
    return rebuild_program(name, origin, tasks, time_blocks, block_ends)


def rebuild_program(
    name: str,
    origin: Callable,
    tasks: list[Task],
    time_blocks: list["TimeBlock"],
    block_ends: set[int],
) -> Program:
    ctr = 0

    def fresh_name() -> str:
        nonlocal ctr
        c = ctr
        ctr += 1
        return f"_checkhint_task{c}"

    def rebuild_task(
        task: Task, live_out: set[str]
    ) -> Tuple[str, list[ast.FunctionDef]]:
        live_in = live_in_task(task, live_out)
        args = ast.arguments(
            [],
            [ast.arg(v, None) for v in live_in],  # arg
            None,  # vararg
            [],  # kwonlyargs
            [],  # kw_defaults
            [],  # kwarg
            [],  # defaults
        )
        ret_stmt = ast.Return(
            ast.Tuple([ast.Name(v, ast.Load()) for v in live_out], ast.Load())
        )
        name = fresh_name()
        # XXX: lots of repeated code here
        match task:
            case Basic(body):
                return (
                    name,
                    [
                        ast.FunctionDef(
                            name,
                            args,
                            body + [ret_stmt],
                            [],  # decorator_list
                            None,  # returns
                        )
                    ],
                )
            case While(test, body, _):
                result = []
                loop_body = []
                for task in body[::-1]:
                    live_in = live_in_task(task, live_out)
                    tname, decls = rebuild_task(task, live_out)
                    result.extend(decls)
                    loop_body.append(
                        [
                            ast.Expr(
                                ast.Call(
                                    ast.Name(tname, ast.Load()),
                                    [ast.Name(v, ast.Load()) for v in live_in],
                                    [],
                                )
                            )
                        ]
                    )
                    live_out = live_in
                while_stmt = ast.While(test, loop_body[::-1], [])
                return (
                    name,
                    result
                    + [
                        ast.FunctionDef(
                            name,
                            args,
                            [while_stmt, ret_stmt],
                            [],
                            None,
                            None,
                            None,
                        )
                    ],
                )
            case For(target, source, body, _):
                result = []
                loop_body = []
                for task in body[::-1]:
                    live_in = live_in_task(task, live_out)
                    tname, decls = rebuild_task(task, live_out)
                    result.extend(decls)
                    loop_body.append(
                        [
                            ast.Expr(
                                ast.Call(
                                    ast.Name(tname, ast.Load()),
                                    [ast.Name(v, ast.Load()) for v in live_in],
                                    [],
                                )
                            )
                        ]
                    )
                    live_out = live_in
                for_stmt = ast.For(target, source, loop_body[::-1], [], None)
                return (
                    name,
                    result
                    + [
                        ast.FunctionDef(
                            name,
                            args,
                            [for_stmt, ret_stmt],
                            [],
                            None,
                        )
                    ],
                )
            case If(test, body, orelse):
                result = []
                then_body = []
                initial_live_in = live_in
                initial_live_out = live_out
                for task in body[::-1]:
                    live_in = live_in_task(task, live_out)
                    tname, decls = rebuild_task(task, live_out)
                    result.extend(decls)
                    then_body.append(
                        [
                            ast.Expr(
                                ast.Call(
                                    ast.Name(tname, ast.Load()),
                                    [ast.Name(v, ast.Load()) for v in live_in],
                                    [],
                                )
                            )
                        ]
                    )
                    live_out = live_in

                orelse_body = []
                live_in = initial_live_in
                live_out = initial_live_out
                for task in orelse[::-1]:
                    live_in = live_in_task(task, live_out)
                    tname, decls = rebuild_task(task, live_out)
                    result.extend(decls)
                    orelse_body.append(
                        [
                            ast.Expr(
                                ast.Call(
                                    ast.Name(tname, ast.Load()),
                                    [ast.Name(v, ast.Load()) for v in live_in],
                                    [],
                                )
                            )
                        ]
                    )
                    live_out = live_in
                if_stmt = ast.If(test, then_body[::-1], orelse_body[::-1])
                return (
                    name,
                    result
                    + [
                        ast.FunctionDef(
                            name,
                            args,
                            [if_stmt, ret_stmt],
                            [],
                            None,
                        )
                    ],
                )
            case With(items, body):
                result = []
                with_body = []
                for task in body[::-1]:
                    live_in = live_in_task(task, live_out)
                    tname, decls = rebuild_task(task, live_out)
                    result.extend(decls)
                    with_body.append(
                        [
                            ast.Expr(
                                ast.Call(
                                    ast.Name(tname, ast.Load()),
                                    [ast.Name(v, ast.Load()) for v in live_in],
                                    [],
                                )
                            )
                        ]
                    )
                    live_out = live_in
                with_stmt = ast.With(items, with_body[::-1], None)
                return (
                    name,
                    result
                    + [
                        ast.FunctionDef(
                            name,
                            args,
                            [with_stmt, ret_stmt],
                            [],
                            None,
                            None,
                            None,
                        )
                    ],
                )
            case t:
                assert_never(t)

    if not tasks:
        raise ValueError("empty task list")
    live_out = initial_live_out(tasks[-1])
    result = []
    result_body = []
    for task in tasks[::-1]:
        live_in = live_in_task(task, live_out)
        tname, decls = rebuild_task(task, live_out)
        result.extend(decls)
        result_body.append(
            ast.Expr(
                ast.Call(
                    ast.Name(tname, ast.Load()),
                    [ast.Name(v, ast.Load()) for v in live_in],
                    [],
                )
            )
        )
        live_out = live_in
    return Program(name, result, result_body, origin)


def max_checkpoints(
    times: list[int], allowance: float, checkpoint_cost: int, failure_prob: float
) -> int:
    return math.ceil(allowance * sum(times) / checkpoint_cost)


@dataclass
class DPResult:
    opt: float
    schedule: frozenset[int]

    def __add__(self, cost: float) -> "DPResult":
        return DPResult(self.opt + cost, self.schedule)

    def __lt__(self, other) -> bool:
        if not isinstance(other, DPResult):
            raise ValueError(f"cannot compare DPResult with `{type(other)}`")
        return self.opt < other.opt


def select_block_ends(
    times: list[int], num_checkpoints: int, failure_prob: float
) -> set[int]:
    @cache
    def t(i: int, j: int) -> int:
        return sum(times[i:j])

    @cache
    def p(i: int, j: int) -> float:
        return 1 - ((1 - failure_prob) ** t(i, j))

    @cache
    def cost(i: int, j: int) -> float:
        p_ = p(i, j)
        if 1 - p_ == 0:
            return math.inf
        return 0.5 * t(i, j) * (p_ / (1 - p_))

    @cache
    def opt(beta: int, j: int) -> DPResult:
        if beta < 1:
            raise ValueError("attempted to schedule fewer than one block")
        if beta == 1:
            return DPResult(cost(0, j), frozenset([j]))
        result = min(opt(beta - 1, i) + cost(i, j) for i in range(beta - 1, j))
        result.schedule |= {j}
        return result

    n = len(times)
    result = opt(min(num_checkpoints, n), n)
    return set(result.schedule)


def cost(body: list[ast.stmt]) -> int:
    return sum(stmt_cost(stmt) for stmt in body)


@dataclass
class TimeBlock:
    t: int
    path: Tuple[int, ...]

    def __add__(self, other: Tuple[int, int]) -> "TimeBlock":
        t_, new_seg = other
        return TimeBlock(self.t + t_, (new_seg,) + self.path)

    def __lt__(self, other) -> bool:
        if not isinstance(other, TimeBlock):
            raise ValueError(f"cannot compare TimeBlock and `{type(other)}`")
        return self.t < other.t


def compute_times(tasks: list[Task]) -> list[TimeBlock]:
    def f(i: int, t: Task) -> list[TimeBlock]:
        match t:
            case Basic(body):
                return [TimeBlock(cost(body), (i,))]
            case While(test, body, count):
                if count > 10_000:
                    count //= 100
                return [
                    c + (stmt_cost(ast.Expr(test)), i) for c in compute_times(body)
                ] * count
            case For(_, _, body, count):
                if count > 10_000:
                    count //= 100
                return [c + (0, i) for c in compute_times(body)] * count
            case If(test, body, orelse):
                body_costs = compute_times(body)
                orelse_costs = compute_times(orelse)
                if len(body_costs) != len(orelse_costs):
                    raise ValueError(
                        "both branches of an [if] statement must have the same "
                        "number of checkpoint hints"
                    )
                return [max(pair) + (0, i) for pair in zip(body_costs, orelse_costs)]
            case With(_, body):
                return [c + (0, i) for c in compute_times(body)]
            case _:
                assert_never(t)

    return [c for i, t in enumerate(tasks) for c in f(i, t)]


# Naively, we assume that all computational operations are of equal weight.
class CostVisitor(ast.NodeVisitor):
    acc: int

    def __init__(self):
        self.acc = 0

    def visit_FunctionDef(self, _node: ast.FunctionDef):
        raise ValueError("cannot analyze liveness of inner functions")

    def visit_AsyncFunctionDef(self, _node: ast.AsyncFunctionDef):
        raise ValueError("cannot analyze liveness of inner functions")

    def visit_ClassDef(self, _node: ast.ClassDef):
        raise ValueError("cannot analyze liveness of inner classes")

    def visit_AsyncFor(self, _node: ast.AsyncFor):
        raise ValueError("async is not yet supported")

    def visit_While(self, node: ast.While):
        initial = self.acc

        for stmt in node.body:
            self.visit(stmt)

        body_cost = self.acc - initial

        if node.body:
            num_iterations = extract_iteration_count(node.body[0])
        else:
            # This is technically incorrect because we actually need to
            # evaluate the loop guard, but w/e
            num_iterations = 0

        self.acc = initial + (body_cost * num_iterations)

    def visit_For(self, node: ast.For):
        initial = self.acc

        for stmt in node.body:
            self.visit(stmt)

        body_cost = self.acc - initial

        if node.body:
            num_iterations = extract_iteration_count(node.body[0])
        else:
            num_iterations = 0

        self.acc = initial + (body_cost * num_iterations)

    def visit_If(self, node: ast.If):
        self.visit(node.test)
        initial = self.acc

        for stmt in node.body:
            self.visit(stmt)
        body_cost = self.acc

        self.acc = initial
        for stmt in node.orelse:
            self.visit(stmt)
        orelse_cost = self.acc

        self.acc = max(body_cost, orelse_cost)

    def visit_BoolOp(self, node: ast.BoolOp):
        super().generic_visit(node)
        self.acc += 1

    def visit_BinOp(self, node: ast.BinOp):
        super().generic_visit(node)
        self.acc += 1

    def visit_UnaryOp(self, node: ast.UnaryOp):
        super().generic_visit(node)
        self.acc += 1

    def visit_IfExp(self, node: ast.IfExp):
        self.visit(node.test)
        initial = self.acc

        self.visit(node.body)
        body_cost = self.acc

        self.acc = initial
        self.visit(node.orelse)
        orelse_cost = self.acc

        self.acc = max(body_cost, orelse_cost)

    def visit_JoinedStr(self, node: ast.JoinedStr):
        super().generic_visit(node)
        self.acc += 1

    def visit_Lambda(self, node: ast.Lambda):
        raise ValueError("cost analysis of lambdas unsupported")


def stmt_cost(stmt: ast.stmt) -> int:
    v = CostVisitor()
    v.visit(stmt)
    return v.acc
