import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba import Mamba, MambaConfig
import argparse
from datetime import datetime
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr
import random
from sklearn.preprocessing import StandardScaler


save_dir = '0728_2330_price/'
use_cuda = True
seed_num = random.randint(0, 999999)
patience = 100
epochs = 1000
# loss_fnc = F.smooth_l1_loss
loss_fnc = None

lr = 0.0005
wd = 1e-5
hidden = 70
layer = 12
n_test = 350
ts_code = 2330
risk_free = 0.017


def evaluation_metric(y_test,y_hat):
    MSE = mean_squared_error(y_test, y_hat)
    RMSE = MSE**0.5
    MAE = mean_absolute_error(y_test,y_hat)
    R2 = r2_score(y_test,y_hat)
    print('%.4f %.4f %.4f %.4f' % (MSE,RMSE,MAE,R2))

def set_seed(seed,cuda):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed(seed)

def dateinf(series, n_test):
    lt = len(series)
    print('Training start',series[0])
    print('Training end',series[lt-n_test-1])
    print('Testing start',series[lt-n_test])
    print('Testing end',series[lt-1])

set_seed(seed_num,use_cuda)


class LogCoshLoss(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, pred, target):
        loss = torch.log(torch.cosh(pred - target))
        return torch.mean(loss)

class CharbonnierLoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps
    def forward(self, pred, target):
        diff = pred - target
        loss = torch.sqrt(diff**2 + self.eps**2)
        return torch.mean(loss)

class QuantileLoss(nn.Module):
    def __init__(self, q=0.5):
        super().__init__()
        self.q = q
    def forward(self, pred, target):
        e = target - pred
        return torch.mean(torch.max(self.q * e, (self.q - 1) * e))



class HybridLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.logcosh = LogCoshLoss()
    def forward(self, pred, target):
        return 0.4 * self.logcosh(pred, target) + 0.6 * F.mse_loss(pred, target)



# loss_fnc = LogCoshLoss()
# loss_fnc = CharbonnierLoss()
# loss_fnc = QuantileLoss()

# loss_fnc = HybridLoss()

class Net(nn.Module):
    def __init__(self, in_dim, out_dim, use_pca=False, pca_dim=None, pca_components=None):
        super().__init__()
        self.use_pca = use_pca

        if self.use_pca:
            assert pca_dim is not None and pca_components is not None, "需提供 pca_dim 和 pca_components"
            self.pca_layer = nn.Linear(in_dim, pca_dim, bias=False)
            self.pca_layer.weight.data = torch.tensor(pca_components, dtype=torch.float32)
            self.pca_layer.weight.requires_grad = False  # 若不希望訓練 PCA 權重
            in_dim = pca_dim  # 更新模型輸入維度

        self.config = MambaConfig(d_model=hidden, n_layers=layer)
        self.mamba = nn.Sequential(
            nn.Linear(in_dim, hidden),
            Mamba(self.config),
            nn.Linear(hidden, out_dim),
            nn.Tanh()
        )

    def forward(self, x):
        if self.use_pca:
            x = self.pca_layer(x)
        x = self.mamba(x)
        return x.flatten()



def PredictWithData(trainX, trainy, testX, save_dir, scaler_y=None, use_pca=False, pca_dim=None, pca_components=None, val_ratio=0.2, patience=patience, loss_fnc=None):
    clf = Net(in_dim=trainX.shape[1], out_dim=1, use_pca=use_pca, pca_dim=pca_dim, pca_components=pca_components)
    

    
    opt = torch.optim.Adam(clf.parameters(),lr=lr,weight_decay=wd)
    
    val_size = int(len(trainX) * val_ratio)
    
    X_train = trainX[:-val_size]
    y_train = trainy[:-val_size]
    X_val = trainX[-val_size:]
    y_val = trainy[-val_size:]
    
    xt = torch.from_numpy(X_train).float().unsqueeze(1)  # → (batch_size, 1, feature_dim)
    xv = torch.from_numpy(X_val).float().unsqueeze(1)
    x_test = torch.from_numpy(testX).float().unsqueeze(1)

    yt = torch.from_numpy(y_train).float()
    yv = torch.from_numpy(y_val).float()
    

    
    if loss_fnc is None:
        loss_fnc = F.mse_loss
    
    best_r2 = -float('inf')
    best_rmse = float('inf')
    best_state_dict = None
    best_epoch = 0

    for e in range(epochs):
        clf.train()
        pred = clf(xt)
        loss = loss_fnc(pred, yt)

        opt.zero_grad()
        loss.backward()
        opt.step()

        clf.eval()
        with torch.no_grad():
            val_pred = clf(xv)
            val_loss = loss_fnc(val_pred, yv)

            val_pred_np = val_pred.cpu().numpy().flatten() if use_cuda else val_pred.numpy().flatten()
            R2 = r2_score(y_val, val_pred_np)
            # RMSE = mean_squared_error(y_val, val_pred_np) ** 0.5

        print(f'Epoch {e:03d} | Val R²: {R2:.4f}')

        if R2 > best_r2:
            best_r2 = R2
            best_epoch = e
            best_state_dict = clf.state_dict()
            best_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")      
        elif e - best_epoch >= patience:
            print(f"Early stopping triggered at epoch {e}")
            break
        
        # print(f'Epoch {e:03d} | Val RMSE: {RMSE:.4f}')

        # if RMSE < best_rmse:
        #     best_rmse = RMSE
        #     best_epoch = e
        #     best_state_dict = clf.state_dict()
        #     best_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")      
        # elif e - best_epoch >= patience:
        #     print(f"Early stopping triggered at epoch {e}")
        #     break

    best_model_path = f"{save_dir}model_{best_timestamp}.pth"
    torch.save(best_state_dict, best_model_path)
    print(f"Best model saved to: {best_model_path}")
    
    
    val_pred_std = val_pred.cpu().numpy().flatten()
    val_true_std = y_val

    # 還原為價格空間
    val_pred_price = scaler_y.inverse_transform(val_pred_std.reshape(-1, 1)).flatten()
    val_true_price = scaler_y.inverse_transform(val_true_std.reshape(-1, 1)).flatten()


    MSE = mean_squared_error(val_true_price, val_pred_price)
    RMSE = MSE ** 0.5
    MAE = mean_absolute_error(val_true_price, val_pred_price)
    R2 = r2_score(val_true_price, val_pred_price)


    # 儲存評估指標
    # 儲存評估指標
    # log_path = f"{save_dir}/evaluation_log.txt"
    # with open(log_path, 'a') as f:
    #     f.write(f"Model: model_{best_timestamp}.pth\n")
    #     f.write(f"  MSE: {MSE:.4f} | RMSE: {RMSE:.4f} | MAE: {MAE:.4f} | R²: {R2:.4f}\n")
    #     f.write("-" * 60 + "\n")
    log_path = f"{save_dir}/evaluation_log.txt"
    with open(log_path, 'a') as f:
        f.write(f"Model: model_{best_timestamp}.pth\n")
        f.write(f"  MSE: {MSE:.4f} | RMSE: {RMSE:.4f} | MAE: {MAE:.4f}\n")
        f.write("-" * 60 + "\n")

    
    clf.load_state_dict(best_state_dict)
    clf.eval()
    mat = clf(x_test)
    if use_cuda: mat = mat.cpu()
    yhat = mat.detach().numpy().flatten()
    yhat = scaler_y.inverse_transform(yhat.reshape(-1, 1)).flatten()
    return yhat


data = pd.read_csv(str(ts_code)+"_value"+'.csv')
data['trade_date'] = pd.to_datetime(data['trade_date'], format='%Y/%m/%d')

# 拆離 label 欄位
close = data.pop('close_TW').values
ratechg = data['close_TW_roc'].values
data.drop(columns=['close_TW_roc'], inplace=True)

# 擷取有效特徵欄位區段
dat = data.iloc[:, 4:].values

scaler_X = StandardScaler()
dat_scaled = scaler_X.fit_transform(dat)

use_pca = False
pca_dim = 25


if use_pca:
    pca = PCA(n_components=pca_dim)
    pca.fit(dat)

    pca_components = pca.components_[:pca_dim]
    dat_pca = dat @ pca_components.T
else:
    dat_pca = dat 
    
trainX, testX = dat_pca[:-n_test], dat_pca[-n_test:]
# trainy = ratechg[:-n_test]
trainy = close[:-n_test]

scaler_y = StandardScaler()
trainy = scaler_y.fit_transform(trainy.reshape(-1, 1)).flatten()

predictions = PredictWithData(trainX, trainy, testX, save_dir, scaler_y, loss_fnc=loss_fnc)
time = data['trade_date'][-n_test:]
data1 = close[-n_test:]
finalpredicted_stock_price = []
pred = close[-n_test-1]
for i in range(n_test):
    pred = close[-n_test-1+i]*(1+predictions[i])
    finalpredicted_stock_price.append(pred)