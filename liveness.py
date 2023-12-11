import ast
from dataclasses import dataclass

from typing import assert_never

from tasks import Task, Basic, While, For, If, With


class Load:
    pass


class Store:
    pass


ExprCtx = Load | Store

def remove_common_ids(s: set[str]) -> set[str]:
    common_names = set(
        [
            "range",
            "np",
            "numpy",
            "scipy",
        ]
    )
    return s - common_names


def initial_live_out(t: Task) -> set[str]:
    def f():
        match t:
            case Basic(body):
                if body and isinstance(body[-1], ast.Return) and body[-1].value:
                    return set(v for v in vars(body[-1].value, Load()))
                else:
                    return set()
            case While(_, body, _) | For(_, _, body, _) | With(_, body):
                if body:
                    return initial_live_out(body[-1])
                else:
                    return set()
            case If(_, body, orelse):
                if body:
                    body_live_out = initial_live_out(body[-1])
                else:
                    body_live_out = set()
                if orelse:
                    orelse_live_out = initial_live_out(orelse[-1])
                else:
                    orelse_live_out = set()
                return body_live_out | orelse_live_out
            case _:
                assert_never(t)
    return remove_common_ids(f())


def live_in_task(t: Task, live_out: set[str]) -> set[str]:
    def f():
        match t:
            case Basic(body):
                return update_liveness(body, live_out)
            case While(test, body, _):
                result = live_out
                for task in body[::-1]:
                    result = live_in_task(task, result)
                return result | vars(test, Load())
            case For(target, source, body, _):
                result = live_out
                for task in body[::-1]:
                    result = live_in_task(task, result)
                result -= vars(target, Store())
                result |= vars(source, Load())
                return result
            case If(test, body, orelse):
                live_in_body = live_out
                for task in body[::-1]:
                    live_in_body = live_in_task(task, live_in_body)
                live_in_orelse = live_out
                for task in orelse[::-1]:
                    live_in_orelse = live_in_task(task, live_in_orelse)
                return live_in_body | live_in_orelse | vars(test, Load())
            case With(items, body):
                result = live_out
                for task in body[::-1]:
                    result = live_in_task(task, result)

                for item in items[::-1]:
                    result |= vars(item.context_expr, Load())
                    if item.optional_vars:
                        result -= vars(item.optional_vars, Store())
                return result
            case _:
                assert_never(t)
    return remove_common_ids(f())


# XXX: Maybe rewrite this to take advantage of the [context] in identifiers
@dataclass
class LivenessVisitor(ast.NodeVisitor):
    live_vars: set[str]

    def visit_FunctionDef(self, _node: ast.FunctionDef):
        raise ValueError("cannot analyze liveness of inner functions")

    def visit_AsyncFunctionDef(self, _node: ast.AsyncFunctionDef):
        raise ValueError("cannot analyze liveness of inner functions")

    def visit_ClassDef(self, _node: ast.ClassDef):
        raise ValueError("cannot analyze liveness of inner classes")

    def visit_Return(self, node: ast.Return):
        if node.value:
            self.live_vars |= vars(node.value, Load())

    def visit_Delete(self, node: ast.Delete):
        for target in node.targets:
            self.live_vars |= vars(target, Load())

    def visit_Assign(self, node: ast.Assign):
        for target in node.targets:
            self.live_vars -= vars(target, Store())
        self.live_vars |= vars(node.value, Load())

    def visit_AugAssign(self, node: ast.AugAssign):
        self.live_vars |= vars(node.value, Load())

    def visit_AnnAssign(self, node: ast.AnnAssign):
        # 'simple' indicates that we annotate simple name without parens
        # params: expr target, expr annotation, expr? value, int simple
        if node.value:
            self.live_vars |= vars(node.value, Load())

    def visit_For(self, node: ast.For):
        # params: expr target, expr iter, stmt* body, stmt* orelse, string? type_comment
        for stmt in node.orelse[::-1]:
            self.visit(stmt)

        for stmt in node.body[::-1]:
            self.visit(stmt)

        self.live_vars -= vars(node.target, Store())
        self.live_vars |= vars(node.iter, Load())

    def visit_AsyncFor(self, node: ast.AsyncFor):
        raise ValueError("async is not yet supported")

    def visit_While(self, node: ast.While):
        # params: expr test, stmt* body, stmt* orelse
        for stmt in node.orelse[::-1]:
            self.visit(stmt)

        for stmt in node.body[::-1]:
            self.visit(stmt)

        self.live_vars |= vars(node.test, Load())

    # XXX: this sucks
    def visit_If(self, node: ast.If):
        # params: expr test, stmt* body, stmt* orelse
        live_vars_on_entry = self.live_vars

        for stmt in node.body[::-1]:
            self.visit(stmt)

        live_vars_into_then = self.live_vars
        self.live_vars = live_vars_on_entry

        for stmt in node.orelse[::-1]:
            self.visit(stmt)

        self.live_vars |= live_vars_into_then

    def visit_With(self, node: ast.With):
        # params: withitem* items, stmt* body, string? type_comment
        for stmt in node.body[::-1]:
            self.visit(stmt)

        for item in node.items[::-1]:
            self.live_vars |= vars(item.context_expr, Load())
            if item.optional_vars:
                self.live_vars -= vars(item.optional_vars, Store())

    def visit_AsyncWith(self, node: ast.AsyncWith):
        raise ValueError("async is not yet supported")

    def visit_Match(self, node: ast.Match):
        # params: expr subject, match_case* cases
        raise ValueError("pattern matching is not yet supported")

    def visit_Raise(self, node: ast.Raise):
        if node.exc:
            self.live_vars |= vars(node.exc, Load())
        if node.cause:
            self.live_vars |= vars(node.cause, Load())

    def visit_Try(self, node: ast.Try):
        raise ValueError("try/except not yet supported")

    def visit_TryStar(self, _node: ast.TryStar):
        # params: stmt* body, excepthandler* handlers, stmt* orelse, stmt* finalbody
        raise ValueError("try/except not yet supported")

    def visit_Assert(self, node: ast.Assert):
        # params: expr test, expr? msg
        self.live_vars |= vars(node.test, Load())
        if node.msg:
            self.live_vars |= vars(node.msg, Load())

    def visit_Import(self, node: ast.Import):
        raise ValueError("cannot analyze liveness for internal imports")

    def visit_ImportFrom(self, node: ast.ImportFrom):
        raise ValueError("cannot analyze liveness for internal imports")

    def visit_Global(self, node: ast.Global):
        # params: identifier* names
        # FIXME: Mutable globals are broken
        self.live_vars -= set(node.names)

    def visit_Nonlocal(self, node: ast.Nonlocal):
        self.live_vars -= set(node.names)

    def visit_Expr(self, node: ast.Expr):
        # params: expr value
        self.live_vars |= vars(node.value, Load())


def update_liveness(stmts: list[ast.stmt], live_out: set[str]) -> set[str]:
    v = LivenessVisitor(live_out)
    for stmt in stmts[::-1]:
        v.visit(stmt)
    return remove_common_ids(v.live_vars)


class VarVisitor(ast.NodeVisitor):
    vars: set[str]
    ctx: ExprCtx

    def __init__(self, ctx: ExprCtx):
        self.vars = set()
        self.ctx = ctx

    def visit_Lambda(self, node: ast.Lambda):
        raise ValueError("lambdas not supported")

    def visit_ListComp(self, node: ast.ListComp):
        raise ValueError("comprehensions are broken")

    def visit_SetComp(self, node: ast.SetComp):
        raise ValueError("comprehensions are broken")

    def visit_DictComp(self, node: ast.DictComp):
        raise ValueError("comprehensions are broken")

    def visit_GeneratorExp(self, node: ast.GeneratorExp):
        raise ValueError("comprehensions are broken")

    def visit_Name(self, node: ast.Name):
        # params: identifier id, expr_context ctx
        match (self.ctx, node.ctx):
            case (Load(), ast.Load()) | (Store(), ast.Store()):
                self.vars.add(node.id)


def vars(exp: ast.expr, ctx: ExprCtx) -> set[str]:
    v = VarVisitor(ctx)
    v.visit(exp)
    return v.vars
