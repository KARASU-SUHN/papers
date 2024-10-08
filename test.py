import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.svm import SVR
import numpy as np
from tqdm import tqdm

class CustomDataset(Dataset):
    def __init__(self, df, phase):
        self.df = df
        self.phase = phase
        self.features = df.drop(columns=['label1', 'label2', 'label3']).values
        self.labels = df[['label1', 'label2', 'label3']].values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        x = self.features[idx]
        y = self.labels[idx]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

class MultiModelTrainer:
    def __init__(self, trainer_cfg, logger):
        self.cfg = trainer_cfg
        self.logger = logger
        self.device = self.cfg.device
        self.amp = self.cfg.amp
        self.epoch = self.cfg.epoch
        self.score_type = self.cfg.score

        # Define models
        self.model = get_model(self.cfg.model)
        self.model = self.model.to(self.device)
        
        # Define loss and optimizer for PyTorch models
        if isinstance(self.model, nn.Module):
            self.criterion = nn.MSELoss().to(self.device)
            self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
            self.scaler = torch.cuda.amp.GradScaler(enabled=self.amp)
        else:
            self.criterion = None
            self.optimizer = None
            self.scaler = None

        self.result = []

    def _load_data(self):
        df = get_dataset(self.cfg.dataset)
        df_train = df[df['fold'] != self.cfg.fold]
        df_eval = df[df['fold'] == self.cfg.fold]

        if self.cfg.phase == 'train':
            self.dataset_train = CustomDataset(df_train, 'train')
            self.dataset_eval = CustomDataset(df_eval, 'eval')
        else:
            self.dataset_train = CustomDataset(df_train, 'eval')
            self.dataset_eval = CustomDataset(df_eval, 'train')

        self.dataloader_train = DataLoader(self.dataset_train, batch_size=self.cfg.dataloader.batch_size, shuffle=True)
        self.dataloader_eval = DataLoader(self.dataset_eval, batch_size=self.cfg.dataloader.batch_size, shuffle=False)

    def _step(self, dataloader: DataLoader, phase: str = 'train', epoch: int = None) -> dict:
        if phase == 'train':
            self.model.train()
        else:
            self.model.eval()

        loss = 0
        y_true = []
        y_pred = []

        num_batches = len(dataloader)

        for data, label in tqdm(dataloader):
            if isinstance(self.model, nn.Module):
                self.optimizer.zero_grad()
                data = data.to(torch.float32).to(self.device)
                label = label.to(torch.float32).to(self.device)

                with torch.set_grad_enabled(phase == 'train'):
                    with torch.cuda.amp.autocast(enabled=(self.amp and (phase == 'train'))):
                        outputs = self.model(data)
                        loss_t = self.criterion(outputs, label)

                if phase == 'train':
                    self.scaler.scale(loss_t).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()

                loss += loss_t.item()
                y_true.extend(list(label.detach().cpu().numpy()))
                y_pred.extend(list(outputs.detach().cpu().numpy()))
            else:
                data = data.cpu().numpy()
                label = label.cpu().numpy()
                if phase == 'train':
                    self.model.fit(data, label)
                predictions = self.model.predict(data)
                y_true.extend(list(label))
                y_pred.extend(list(predictions))

        loss = loss / num_batches
        score = self.get_score(y_pred, y_true)

        return score, loss

    def train(self):
        self._load_data()
        dl_dct = {"train": self.dataloader_train, "eval": self.dataloader_eval}

        for epoch in range(self.epoch):
            print(f"--------------------------------------")
            print(f"Epoch {epoch + 1}")

            for phase in ["train", "eval"]:
                score, loss = self._step(dl_dct[phase], phase, epoch)
                print(f"loss/{phase}:{loss}")
                print(f"score/{phase}:{score}")
                self.logger.log_metrics({f"loss/{phase}": loss}, step=epoch)
                self.logger.log_metrics({f"metrics/{phase}": score}, step=epoch)

            if self.is_best_model(self.best_score, score):
                print("best model ever!")
                self.logger.log_model(model=self.model, model_name="model_best")

        self.logger.log_model(model=self.model, model_name="model_last")

def get_trainer(trainer_cfg=trainer_cfg(), logger=Logger()):
    return MultiModelTrainer(trainer_cfg, logger)






model/



import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.svm import SVR
from torch.utils.data import DataLoader
import timm


class TimmRegressor(nn.Module):
    def __init__(self, backbone, out_dim, in_chans,pre_train=True):
        super(TimmRegressor, self).__init__()

        # for timm
        self.backbone =  timm.create_model(
            backbone, 
            pretrained=pre_train, 
            num_classes= 0,
            in_chans=in_chans
        )
        in_features = self.backbone.num_features

        self.head = nn.Sequential(
            nn.Linear(in_features, round(in_features/2)),
            nn.ReLU(),
            nn.Linear(round(in_features/2), out_dim)
        )

    def forward(self, x):
        embedding = self.backbone(x)
        y = self.head(embedding)
        self.embedding = embedding

        return y
    

# RandomForestRegressor class
class RandomForestRegressorModel:
    def __init__(self, n_estimators=100):
        self.model = RandomForestRegressor(n_estimators=n_estimators)

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

# LSTM model
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(1, x.size(0), hidden_size).to(x.device)
        c0 = torch.zeros(1, x.size(0), hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

# XGBoost model
class XGBoostModel:
    def __init__(self, n_estimators=100):
        self.model = xgb.XGBRegressor(n_estimators=n_estimators)

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

# SVM model
class SVMModel:
    def __init__(self):
        self.model = SVR()

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

# MLP model
class MLPModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLPModel, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        return self.model(x)

# Function to get the model based on configuration
def get_model(model_cfg):
    if model_cfg.name == "TimmRegressor":
        model = TimmRegressor(
            backbone=model_cfg.backbone, 
            pre_train=model_cfg.pre_train,
            in_chans=model_cfg.in_chans,
            out_dim=model_cfg.out_dim
        )
    elif model_cfg.name == "RandomForestRegressor":
        model = RandomForestRegressorModel(n_estimators=model_cfg.n_estimators)
    elif model_cfg.name == "LSTMModel":
        model = LSTMModel(
            input_size=model_cfg.input_size, 
            hidden_size=model_cfg.hidden_size, 
            output_size=model_cfg.output_size, 
            num_layers=model_cfg.num_layers
        )
    elif model_cfg.name == "XGBoostModel":
        model = XGBoostModel(n_estimators=model_cfg.n_estimators)
    elif model_cfg.name == "SVMModel":
        model = SVMModel()
    elif model_cfg.name == "MLPModel":
        model = MLPModel(
            input_size=model_cfg.input_size, 
            hidden_size=model_cfg.hidden_size, 
            output_size=model_cfg.output_size
        )
    else:
        raise Exception(f'{model_cfg.name} is not implemented')    

    return model
