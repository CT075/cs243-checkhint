import ast
import inspect

from task_splitter import split_tasks
from checkpoints import configure_tasks_by_checkpoint


def schedule_checkpoints(
    allowance: float = 0.1,
    checkpoint_cost: int = 10,
    failure_prob: float = 0.1,
    checkpoint_fn_name="_checkpoint",
):
    def wrapper(f):
        name, tasks = split_tasks(f, checkpoint_fn_name)
        return configure_tasks_by_checkpoint(name, f, tasks, allowance, checkpoint_cost, failure_prob)
    return wrapper

def _checkpoint():
    pass
