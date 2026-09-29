"""稠密直接解法（部分主元高斯消元），仅用于与 CG 迭代解对拍验证。"""


def solve_dense(rows, b):
    """高斯消元（部分主元）解 Ax = b，返回解向量。奇异矩阵抛 ValueError。"""
    n = len(rows)
    if n == 0 or any(len(row) != n for row in rows) or len(b) != n:
        raise ValueError("维度不匹配")
    # 增广矩阵，原地消元
    mat = [list(rows[i]) + [float(b[i])] for i in range(n)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(mat[r][col]))
        if abs(mat[piv][col]) < 1e-300:
            raise ValueError("矩阵奇异，直接法失败")
        mat[col], mat[piv] = mat[piv], mat[col]
        inv = 1.0 / mat[col][col]
        for r in range(col + 1, n):
            f = mat[r][col] * inv
            if f != 0.0:
                for c in range(col, n + 1):
                    mat[r][c] -= f * mat[col][c]
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        s = mat[i][n] - sum(mat[i][j] * x[j] for j in range(i + 1, n))
        x[i] = s / mat[i][i]
    return x
