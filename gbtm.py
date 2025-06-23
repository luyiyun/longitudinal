from dataclasses import dataclass

import pandas as pd
import numpy as np
import scipy.stats as sst
from statsmodels.regression.linear_model import WLS
from scipy.special import logsumexp
from tqdm import tqdm
import matplotlib.pyplot as plt
from patsy import dmatrices, dmatrix


@dataclass
class GBTM:
    """
    基于EM算法的基于组的轨迹模型 (Group-Based Trajectory Model)，使用dataclass和公式API。

    该实现假设结局变量服从正态分布。

    参数:
    ----------
    n_components : int
        潜类别的数量 (G)。
    formula : str
        一个patsy风格的公式，用于定义轨迹的固定效应部分。
        例如: 'y ~ 1 + time + I(time**2)'
    max_iter : int, optional (default=100)
        EM算法的最大迭代次数。
    tol : float, optional (default=1e-6)
        收敛的阈值。
    random_state : int, optional (default=None)
        用于初始化聚类的随机种子。
    """

    n_components: int
    formula: str
    group: str | None = None  # 用于指定分组变量，默认为None, 即使用index作为分组变量
    max_iter: int = 500
    tol: float = 1e-10
    random_state: int = None

    def __post_init__(self):
        if self.n_components <= 0:
            raise ValueError("n_components 必须大于 0。")

        self._rng = np.random.default_rng(self.random_state)

    def _initialize_parameters(self, y: np.ndarray, X: np.ndarray):
        """初始化模型参数"""
        self.pi_log_ = self._rng.standard_normal(self.n_components)
        # self.betas_ = self._rng.standard_normal((self.n_components, n_betas))
        # self.sigmas_ = self._rng.uniform(size=self.n_components)

        wls_model = WLS(y, X)
        wls_results = wls_model.fit()
        self.betas_ = (
            np.stack([wls_results.params] * self.n_components, axis=0)
            + self._rng.standard_normal((self.n_components, wls_results.params.size))
            * 0.1
        )
        self.sigmas_ = np.full(
            self.n_components, np.sqrt((wls_results.resid**2).mean())
        )

    def _e_step(
        self, y: np.ndarray, X: np.ndarray, group: np.ndarray
    ) -> tuple[pd.DataFrame, float]:
        mu = X @ self.betas_.T  # (n_samples, n_components)
        log_pdf = sst.norm.logpdf(
            y,
            loc=mu,
            scale=self.sigmas_,
        )
        log_prob_ind = (
            pd.DataFrame(log_pdf, index=group, columns=np.arange(self.n_components))
            .groupby(level=0)
            .sum()
        )
        log_prob_ind += self.pi_log_
        log_likelihood_ind = logsumexp(log_prob_ind.values, axis=1)
        posteriors = pd.DataFrame(
            np.exp(log_prob_ind - log_likelihood_ind[:, None]), index=log_prob_ind.index
        )
        return posteriors, log_likelihood_ind.sum()

    def _m_step(
        self, y: np.ndarray, X: np.ndarray, posteriors: pd.DataFrame, group: np.ndarray
    ):
        """M-步: 更新参数"""

        self.pi_log_ = np.log(posteriors.values.mean(axis=0))
        for g in range(self.n_components):
            weights = posteriors.loc[group, g]
            wls_model = WLS(y, X, weights=weights)
            wls_results = wls_model.fit()
            self.betas_[g, :] = wls_results.params

            weighted_ssr = np.sum(weights * (wls_results.resid**2))
            sum_weights = np.sum(weights)
            self.sigmas_[g] = (
                np.sqrt(weighted_ssr / sum_weights) if sum_weights > 0 else np.inf
            )

    def fit(self, data: pd.DataFrame) -> "GBTM":
        """
        拟合GBTM模型。

        参数:
        ----------
        data : pd.DataFrame
            包含因变量、时间变量和ID的DataFrame。
            必须包含公式中用到的所有列名，以及一个'id'列。
        """
        # 1. 初始化
        # 复制数据以避免修改原始DataFrame
        fit_data = data.copy()
        y_patsy, X_patsy = dmatrices(
            self.formula, data=fit_data, return_type="dataframe"
        )
        group = (
            fit_data.loc[y_patsy.index, self.group].values
            if self.group is not None
            else y_patsy.index.values
        )
        self._initialize_parameters(y_patsy.values, X_patsy.values)

        # 2. EM迭代
        self.history_ = []
        prev_llk = -np.inf
        for i in tqdm(range(self.max_iter)):
            posteriors, llk = self._e_step(y_patsy.values, X_patsy.values, group)
            self._m_step(y_patsy.values, X_patsy.values, posteriors, group)
            self.history_.append(llk)

            if i > 0 and np.abs(llk - prev_llk) < self.tol:
                break

            prev_llk = llk
        else:
            print(f"警告: 模型在达到最大迭代次数 {self.max_iter} 后仍未收敛。")

        self.params_ = pd.DataFrame(
            self.betas_,
            columns=X_patsy.columns,
            index=[f"class_{g}" for g in range(self.n_components)],
        )

        df_posterior, _ = self._e_step(y_patsy.values, X_patsy.values, group)
        df_posterior.rename(columns=lambda x: f"prob_{x}", inplace=True)
        df_posterior["group"] = df_posterior.idxmax(axis=1).str.replace(
            "prob_", "class_"
        )
        self.posterior_summary_ = df_posterior.groupby("group").mean()

        return self

    def predict_trajectory(self, data_new: pd.DataFrame, var_name: str) -> pd.DataFrame:
        """为新的数据点预测每个类别的轨迹值"""
        X_new = dmatrix(
            self.formula.split("~")[-1], data=data_new, return_type="dataframe"
        )

        trajectories = {}
        for g in range(self.n_components):
            trajectories[f"class_{g}"] = X_new.values @ self.betas_[g, :]

        res = pd.DataFrame(trajectories, index=data_new.index)
        res[var_name] = data_new.loc[X_new.index, var_name]
        return res


if __name__ == "__main__":
    df = pd.read_csv("./data/paquid.csv", index_col=0)
    df["age65"] = (df["age"] - 65) / 10
    print(df.head())

    # 2. 实例化并拟合模型 (注意调用方式的变化)
    # 现在只需要传入包含所有信息的DataFrame
    gbtm = GBTM(
        n_components=4, formula="MMSE ~ 1 + age65 + I(age65**2)", random_state=42
    )
    gbtm.fit(df)
    print(gbtm.params_)
    print(gbtm.posterior_summary_)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(gbtm.history_)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Log Likelihood")
    fig.savefig("./results/py_gbtm_history.png")

    data_new = pd.DataFrame(
        {
            "age": np.linspace(60, 100, 100),
        }
    )
    data_new["age65"] = (data_new["age"] - 65) / 10
    predictions = gbtm.predict_trajectory(data_new, var_name="age")
    print(predictions.head())

    fig, ax = plt.subplots(figsize=(10, 6))
    for i in range(gbtm.n_components):
        ax.plot(
            predictions["age"],
            predictions[f"class_{i}"],
            label=f"Class {i}",
        )
    ax.set_xlabel("age")
    ax.set_ylabel("MMSE")
    ax.legend()
    fig.savefig("./results/py_gbtm_predictions.png")
