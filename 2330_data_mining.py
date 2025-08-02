import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def scatterplot_features(X, y, feature_names=None, save_dir=None):
    """
    Plot scatterplots of each X_i vs y

    Parameters:
        X: ndarray of shape (n_samples, n_features)
        y: ndarray of shape (n_samples,)
        feature_names: optional list of feature names
        save_dir: optional directory to save plots
    """
    n_features = X.shape[1]
    
    for i in range(n_features):
        plt.figure(figsize=(5,4))
        plt.scatter(X[:, i], y, alpha=0.5)
        fname = f"Feature {i}" if not feature_names else feature_names[i]
        plt.title(f"{fname} vs Target")
        plt.xlabel(fname)
        plt.ylabel("Target (y)")
        plt.grid(True)

        if save_dir:
            plt.savefig(f"{save_dir}/scatter_{fname}.png", dpi=150)
            plt.close()
        else:
            plt.show()


ts_code=2330

data = pd.read_csv(str(ts_code)+"_value"+'.csv')
data['trade_date'] = pd.to_datetime(data['trade_date'], format='%Y/%m/%d')

# 拆離 label 欄位
close = data.pop('close_TW').values
ratechg = data['close_TW_roc'].values
data.drop(columns=['close_TW_roc'], inplace=True)

X = data.iloc[:, 4:].values  # shape = (n_samples, n_features)
y = ratechg      # y = close_TW 的漲跌百分比
feature_names = data.columns.tolist()

scatterplot_features(X, y, feature_names=feature_names)

