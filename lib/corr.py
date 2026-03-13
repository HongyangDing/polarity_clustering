import torch
import torch.nn.functional as F


def correlate_conv1d_matrix(data1, temp, mode="full"):
    """
    使用 torch.nn.functional.conv1d 计算一维矩阵模板的归一化互相关。

    参数:
    - data1: (B, N) 一条或多条模板波形
    - temp: (C, M) 多条待比对波形
    - mode: 目前主线只使用 "full"

    返回:
    - shape (B, C, L) 的互相关矩阵
    """
    if mode != "full":
        raise ValueError("Only mode='full' is supported in the main pipeline.")

    if data1.ndim != 2 or temp.ndim != 2:
        raise ValueError("data1 and temp must both be 2-D matrices.")

    data1 = data1 - data1.mean(dim=1, keepdims=True)
    temp = temp - temp.mean(dim=1, keepdims=True)
    denominator = (
        torch.sqrt(torch.sum(data1**2, dim=1, keepdim=True))
        * torch.sqrt(torch.sum(temp**2, dim=1)).reshape(1, -1)
        + 1e-6
    )

    data1 = data1.unsqueeze(1)
    temp = temp.unsqueeze(1)
    output = F.conv1d(data1, temp, padding=temp.shape[-1] - 1)
    return output / denominator.unsqueeze(-1)
