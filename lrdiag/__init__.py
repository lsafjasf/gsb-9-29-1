"""lrdiag：学习率调度、训练诊断与发散自愈（纯标准库）。"""

from .schedulers import (
    LRScheduler,
    StepDecay,
    CosineDecay,
    PlateauDecay,
    build_scheduler,
)
from .diagnostics import (
    LossDiagnoser,
    Diagnosis,
    DIVERGING,
    PLATEAU,
    OSCILLATING,
    HEALTHY,
)
from .trainer import GDTrainer, RollbackEvent, RunResult, assert_rollback
from .objectives import Quadratic, Rosenbrock

__all__ = [
    "LRScheduler",
    "StepDecay",
    "CosineDecay",
    "PlateauDecay",
    "build_scheduler",
    "LossDiagnoser",
    "Diagnosis",
    "DIVERGING",
    "PLATEAU",
    "OSCILLATING",
    "HEALTHY",
    "GDTrainer",
    "RollbackEvent",
    "RunResult",
    "assert_rollback",
    "Quadratic",
    "Rosenbrock",
]
