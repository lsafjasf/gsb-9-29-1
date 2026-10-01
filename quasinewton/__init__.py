"""拟牛顿法小库（纯标准库）：BFGS + 强 Wolfe/Armijo 线搜索。"""

from .bfgs import minimize_bfgs, minimize_gd, OptimizeResult
from .linesearch import line_search, LineSearchResult

__all__ = ["minimize_bfgs", "minimize_gd", "line_search",
           "OptimizeResult", "LineSearchResult"]
