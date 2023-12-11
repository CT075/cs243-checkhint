import ast
import inspect
from collections.abc import Iterator
from dataclasses import dataclass, field

from typing import Tuple, assert_never

from tasks import Task, Basic, While, For, If, With


def split_tasks(func, checkpoint_fn_name: str = '_checkpoint') -> Tuple[str, list[Task]]:
    body = ast.parse(inspect.getsource(func))
    if isinstance(body, ast.Module):
        if len(body.body) != 1:
            raise ValueError("internal error: was [schedule_checkpoints] called manually?")
        if not isinstance(body.body[0], ast.FunctionDef):
            raise ValueError("split_tasks must be called on a function")
    else:
        raise ValueError("internal error: was [schedule_checkpoints] called manually?")
    name = body.body[0].name
    events = mark_events(body, checkpoint_fn_name)
    return name, collect_tasks(events)


@dataclass
class Checkpoint:
    pass


@dataclass
class StartElse:
    pass


@dataclass
class End:
    pass


@dataclass
class Stmt:
    v: ast.stmt


@dataclass
class StartWhile:
    test: ast.expr
    num_iterations: int


@dataclass
class StartFor:
    target: ast.expr
    source: ast.expr
    num_iterations: int


@dataclass
class StartIf:
    test: ast.expr


@dataclass
class StartWith:
    items: list[ast.withitem]


Event = (
    Checkpoint | StartElse | End | Stmt | StartWhile | StartFor | StartIf | StartWith
)


def extract_iteration_count(node: ast.stmt) -> int:
    if not isinstance(node, ast.AnnAssign):
        raise ValueError(
            "the first line of a loop must be annotated with the line `_iterations: n`"
        )
    if not isinstance(node.target, ast.Name):
        raise ValueError(
            "the first line of a loop must be annotated with the line `_iterations: n`"
        )
    if node.target.id != "_iterations":
        raise ValueError(
            "the first line of a loop must be annotated with the line `_iterations: n`"
        )
    if not node.simple:
        raise ValueError(
            "the first line of a loop must be annotated with the line `_iterations: n`"
        )
    if not isinstance(node.annotation, ast.Constant):
        raise ValueError("the iteration counter must be a constant")
    if not isinstance(node.annotation.value, int):
        raise ValueError("the iteration counter must be an int")
    return node.annotation.value


@dataclass
class Walker(ast.NodeVisitor):
    checkpoint_name: str
    events: list[Event] = field(default_factory=list)

    def emit(self, ev: Event):
        self.events.append(ev)

    def emit_statement(self, node: ast.stmt):
        self.emit(Stmt(node))

    def visit_Expr(self, node: ast.Expr):
        match node.value:
            case ast.Call(ast.Name(id, _), [], []) if id == self.checkpoint_name:
                self.emit(Checkpoint())
            case _:
                self.emit_statement(node)

    def visit_While(self, node: ast.While):
        if node.body:
            num_iterations = extract_iteration_count(node.body[0])
        else:
            num_iterations = 0
        self.emit(
            StartWhile(
                node.test,
                num_iterations,
            )
        )
        for child in node.body[1:]:
            self.visit(child)
        if node.orelse:
            self.emit(StartElse())
        for child in node.orelse:
            self.visit(child)
        self.emit(End())

    def visit_If(self, node: ast.If):
        self.emit(StartIf(node.test))
        for child in node.body:
            self.visit(child)
        if node.orelse:
            self.emit(StartElse())
        for child in node.orelse:
            self.visit(child)
        self.emit(End())

    def visit_For(self, node: ast.For):
        if node.body:
            num_iterations = extract_iteration_count(node.body[0])
        else:
            num_iterations = 0
        self.emit(StartFor(node.target, node.iter, num_iterations))
        for child in node.body:
            self.visit(child)
        if node.orelse:
            self.emit(StartElse())
        for child in node.orelse:
            self.visit(child)
        self.emit(End())

    def visit_AsyncFor(self, _node: ast.AsyncFor):
        raise ValueError("analysis of async functions not supported")

    def visit_With(self, node: ast.With):
        self.emit(StartWith(node.items))
        for child in node.body:
            self.visit(child)
        self.emit(End())

    def visit_FunctionDef(self, node: ast.FunctionDef):
        for child in node.body:
            self.visit(child)

    def visit_AsyncFunctionDef(self, _node: ast.AsyncFunctionDef):
        raise ValueError("analysis of async functions not supported")

    def generic_visit(self, node: ast.AST):
        if isinstance(node, ast.stmt):
            self.emit_statement(node)
        else:
            super().generic_visit(node)


def mark_events(ast: ast.AST, checkpoint_fn_name: str) -> list[Event]:
    visitor = Walker(checkpoint_fn_name)
    visitor.visit(ast)
    return visitor.events


@dataclass
class Tasks:
    tasks: list[Task]


@dataclass
class Ended:
    pass


@dataclass
class SawElse:
    pass


@dataclass
class EOF:
    pass


ConsumeResult = Basic | Tasks
EndReason = Ended | SawElse | EOF


class State:
    accum: list[Task]
    current_block: list[ast.stmt]

    done: bool
    end_reason: EndReason

    def __init__(self):
        self.current_block = []
        self.accum = []
        self.done = False
        self.end_reason = EOF()

    def push(self, s: ast.stmt) -> None:
        if self.done:
            raise ValueError("called [push] on [done] state")
        self.current_block.append(s)

    def split(self) -> None:
        if self.done:
            raise ValueError("called [split] on [done] state")
        self.accum.append(Basic(self.current_block))
        self.current_block = []

    def finalize(self) -> None:
        self.accum.append(Basic(self.current_block))
        self.done = True

    def set_end(self) -> None:
        self.end_reason = Ended()
        self.finalize()

    def set_else(self) -> None:
        self.end_reason = SawElse()
        self.finalize()

    def decompose(self) -> Tuple[ConsumeResult, EndReason]:
        if not self.done:
            raise ValueError("called [decompose] on not [done] state")
        match self.accum:
            case [Basic(_) as basic]:
                return (basic, self.end_reason)
            case _:
                return (Tasks(self.accum), self.end_reason)

    def push_task(self, task: Task) -> None:
        if self.current_block:
            self.split()
        self.accum.append(task)

    def push_tasks(self, tasks: list[Task]) -> None:
        for task in tasks:
            self.push_task(task)


def consume_block(evts: Iterator[Event]) -> Tuple[ConsumeResult, EndReason]:
    state = State()

    # This iteration relies on the fact that [evt] shares its state across
    # recursive invocations of [consume_block], and that the next item yielded
    # after returning from a recursive [consume_block] will come after the
    # token that caused us to exit.
    for evt in evts:
        match evt:
            case StartWhile(test, num_iterations):
                stmts, end_reason = consume_block(evts)
                match end_reason:
                    case Ended():
                        orelse_stmts: ConsumeResult = Basic([])
                    case SawElse():
                        orelse_stmts, end_reason_ = consume_block(evts)
                        match end_reason_:
                            case Ended():
                                pass
                            case SawElse():
                                raise ValueError("duplicated [Else]")
                            case EOF():
                                raise ValueError("unclosed [Else]")
                            case _:
                                assert_never(end_reason_)
                    case EOF():
                        raise ValueError("unclosed [While]")
                    case _:
                        assert_never(end_reason)
                match (stmts, orelse_stmts):
                    case (Basic(body), Basic(orelse)):
                        state.push(ast.While(test, body, orelse))
                    case (Basic(body), Tasks(orelse)):
                        state.push(ast.While(test, body, []))
                        state.push_tasks(orelse)
                    case (Tasks(body), Basic(orelse)):
                        state.push_task(While(test, body, num_iterations))
                        for stmt in orelse:
                            state.push(stmt)
                    case (Tasks(body), Tasks(orelse)):
                        state.push_task(While(test, body, num_iterations))
                        state.push_tasks(orelse)
                    case _:
                        assert False
            case StartFor(target, source, num_iterations):
                stmts, end_reason = consume_block(evts)
                match end_reason:
                    case Ended():
                        orelse_stmts = Basic([])
                    case SawElse():
                        orelse_stmts, end_reason_ = consume_block(evts)
                        match end_reason_:
                            case Ended():
                                pass
                            case SawElse():
                                raise ValueError("duplicated [Else]")
                            case EOF():
                                raise ValueError("unclosed [Else]")
                            case _:
                                assert_never(end_reason_)
                    case EOF():
                        raise ValueError("unclosed [For]")
                    case _:
                        assert_never(end_reason)
                match (stmts, orelse_stmts):
                    case (Basic(body), Basic(orelse)):
                        state.push(ast.For(target, source, body, orelse))
                    case (Basic(body), Tasks(orelse)):
                        state.push(ast.For(target, source, body, []))
                        state.push_tasks(orelse)
                    case (Tasks(body), Basic(orelse)):
                        state.push_task(For(target, source, body, num_iterations))
                        for stmt in orelse:
                            state.push(stmt)
                    case (Tasks(body), Tasks(orelse)):
                        state.push_task(For(target, source, body, num_iterations))
                        state.push_tasks(orelse)
                    case _:
                        assert False
            case StartIf(test):
                stmts, end_reason = consume_block(evts)
                match end_reason:
                    case Ended():
                        orelse_stmts = Basic([])
                    case SawElse():
                        orelse_stmts, end_reason_ = consume_block(evts)
                        match end_reason_:
                            case Ended():
                                pass
                            case SawElse():
                                raise ValueError("duplicated [Else]")
                            case EOF():
                                raise ValueError("unclosed [Else]")
                            case _:
                                assert_never(end_reason_)
                    case EOF():
                        raise ValueError("unclosed [If]")
                    case _:
                        assert_never(end_reason)
                match (stmts, orelse_stmts):
                    case (Basic(body), Basic(orelse)):
                        state.push(ast.If(test, body, orelse))
                    case (Basic(body), Tasks(orelse)):
                        state.push(ast.If(test, body, []))
                        state.push_tasks(orelse)
                    case (Tasks(body), Basic(orelse)):
                        state.push_task(If(test, body, [Basic(orelse)]))
                    case (Tasks(body), Tasks(orelse)):
                        state.push_task(If(test, body, orelse))
                    case _:
                        assert False
            case StartWith(items):
                stmts, end_reason = consume_block(evts)
                match end_reason:
                    case Ended():
                        pass
                    case SawElse():
                        raise ValueError("[With] closed by [Else]")
                    case EOF():
                        raise ValueError("unclosed [With]")
                    case _:
                        assert_never(end_reason)
                match stmts:
                    case Basic(body):
                        state.push(ast.With(items, body))
                    case Tasks(body):
                        state.push_task(With(items, body))
            case Checkpoint():
                state.split()
            case StartElse():
                state.set_else()
                break
            case End():
                state.set_end()
                break
            case Stmt(stmt):
                state.push(stmt)
            case _:
                assert_never(evt)
    else:
        state.finalize()

    return state.decompose()


def collect_tasks(evts: list[Event]) -> list[Task]:
    stmts, end_reason = consume_block(iter(evts))
    match end_reason:
        case EOF():
            pass
        case Ended():
            raise ValueError("unmatched [End]")
        case SawElse():
            raise ValueError("unmatched [Else]")
        case _:
            assert_never(end_reason)
    match stmts:
        case Tasks(tasks):
            return tasks
        case Basic(_) as basic:
            return [basic]
        case _:
            assert_never(stmts)
