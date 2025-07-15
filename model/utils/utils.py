import torch, random, os
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from hyperimpute.plugins.utils.simulate import simulate_nan
import pandas as pd

def enable_reproducible_results(seed: int = 0) -> None:
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def nanmean(v, *args, **kwargs):
    """
    A Pytorch version on Numpy's nanmean
    """
    v = v.clone()
    is_nan = torch.isnan(v)
    v[is_nan] = 0
    return v.sum(*args, **kwargs) / (~is_nan).float().sum(*args, **kwargs)


def ampute(x, mechanism, p_miss):
    x_simulated = simulate_nan(np.asarray(x), p_miss, mechanism)

    mask = x_simulated["mask"]
    x_miss = x_simulated["X_incomp"]

    return pd.DataFrame(x), pd.DataFrame(x_miss), pd.DataFrame(mask)


def scale_data(X):
    """
    对数据进行标准化处理。

    将输入的数据转换为numpy数组，并使用StandardScaler进行标准化处理。
    StandardScaler通过移除均值和缩放数据到单位方差来标准化特征。

    注释掉的代码提供了替代方案，例如使用MinMaxScaler进行最小-最大缩放，
    以及显示地调用fit方法，这些可以根据实际需求选择使用。

    参数:
    X -- 输入数据，可以是列表、numpy数组等。

    返回:
    经过标准化处理的数据，以numpy数组形式返回。
    """
    # 将输入数据转换为numpy数组，确保数据格式兼容性
    X = np.asarray(X)

    # 实例化StandardScaler，用于数据标准化
    preproc = StandardScaler()

    # 使用StandardScaler对数据进行拟合和转换
    # fit方法被注释掉，直接使用fit_transform方法进行拟合和转换
    # preproc.fit(X)

    # 返回经过标准化处理的数据
    return np.asarray(preproc.fit_transform(X))


def diff_scale_data(X):
    X = np.asarray(X)
    # preproc = MinMaxScaler()
    preproc = MinMaxScaler()
    print(f"we are using minmax scaler for diffusion models!")
    # preproc.fit(X)
    return np.asarray(preproc.fit_transform(X))

def simulate_scenarios(X, mechanisms=["MAR", "MNAR", "MCAR"], percentages=[0.1, 0.3, 0.5, 0.7], diff_model=False):
    """
    模拟不同的缺失数据场景。

    该函数通过应用不同的缺失数据机制和百分比来模拟输入数据集X的不同缺失数据场景。
    支持的缺失数据机制包括MAR、MNAR和MCAR，并允许在四个不同的缺失百分比下进行模拟。

    参数:
    - X: 输入数据集，矩阵形式。
    - mechanisms: 包含缺失数据机制字符串的列表，默认为["MAR", "MNAR", "MCAR"]。
    - percentages: 包含缺失数据比例的列表，默认为[0.1, 0.3, 0.5, 0.7]。
    - diff_model: 布尔值，指示是否使用不同的数据缩放方法，默认为False。

    返回:
    - datasets: 包含在不同缺失机制和百分比下模拟的数据集的字典。
    """
    # 根据diff_model的值选择不同的数据缩放方法
    X = scale_data(X) if not diff_model else diff_scale_data(X)
    datasets = {}

    # 遍历每种缺失机制和每个缺失百分比以模拟缺失数据场景
    for ampute_mechanism in mechanisms:
        for p_miss in percentages:
            # 如果当前缺失机制不在datasets字典中，则添加
            if ampute_mechanism not in datasets:
                datasets[ampute_mechanism] = {}

            # 在当前机制和百分比下模拟缺失数据，并存储结果
            datasets[ampute_mechanism][p_miss] = ampute(X, ampute_mechanism, p_miss)

    return datasets



# third party
import numpy as np


def MAE(X: np.ndarray, X_true: np.ndarray, mask: np.ndarray, verbose=0) -> np.ndarray:
    """
    Mean Absolute Error (MAE) between imputed variables and ground truth.

    Args:
        X : Data with imputed variables.
        X_true : Ground truth.
        mask : Missing value mask (missing if True)

    Returns:
        MAE : np.ndarray
    """
    mask_ = mask.astype(bool)
    if verbose == 0:
        return np.absolute(X[mask_] - X_true[mask_]).sum() / mask_.sum()
    else:
        num_miss = mask_.sum(axis=0)
        output = []
        for i in range(X.shape[-1]):
            if num_miss[i] == 0:
                output += ['0']
            else:
                _output = np.absolute(X[:, i][mask_[:, i]]-X_true[:, i][mask_[:, i]])
                _output = _output.sum() / mask_[:, i].sum()
                output += [str(_output.round(5))]

        return output




__all__ = ["MAE"]
