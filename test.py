import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, df, phase, features, target):
        self.df = df
        self.phase = phase
        self.features = df[features].values
        self.labels = df[target].values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        x = self.features[idx]
        y = self.labels[idx]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

custom_dataset_dict = {
    "CustomDataset": CustomDataset
}

def get_dataset(dataset_cfg, phase):
    if dataset_cfg.name in custom_dataset_dict.keys():
        dataset = custom_dataset_dict[dataset_cfg.name](
            df=dataset_cfg.df,
            phase=phase,
            features=dataset_cfg.features,
            target=dataset_cfg.target
        )
        return dataset
    else:
        raise ValueError(f"Dataset {dataset_cfg.name} is not implemented")

def get_dataloader(dataset, dataloader_cfg, phase):
    return DataLoader(
        dataset,
        batch_size=dataloader_cfg.batch_size,
        shuffle=(phase == 'train'),
        num_workers=dataloader_cfg.num_workers
    )

from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.svm import SVR
import torch.nn as nn

class RandomForestModel:
    def __init__(self, n_estimators=100):
        self.model = RandomForestRegressor(n_estimators=n_estimators)

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

class XGBoostModel:
    def __init__(self, n_estimators=100):
        self.model = xgb.XGBRegressor(n_estimators=n_estimators)

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

class SVMModel:
    def __init__(self):
        self.model = SVR()

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

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

def get_model(model_cfg):
    if model_cfg.name == "RandomForestModel":
        model = RandomForestModel(n_estimators=model_cfg.n_estimators)
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
        raise ValueError(f"Model {model_cfg.name} is not implemented")
    return model



class MultiModelTrainer:
    def __init__(self, trainer_cfg, logger):
        self.cfg = trainer_cfg
        self.logger = logger
        self.device = self.cfg.device
        self.epoch = self.cfg.epoch
        self.score_type = self.cfg.score

        # Define models
        self.model = get_model(self.cfg.model)
        
        if isinstance(self.model, nn.Module):
            self.model = self.model.to(self.device)
            self.criterion = nn.MSELoss().to(self.device)
            self.optimizer, self.scheduler, self.model = get_optimizer(
                self.cfg.optimizer, 
                self.model,
                self.cfg.epoch
            )
            self.scaler = torch.cuda.amp.GradScaler(enabled=self.amp)
        else:
            self.criterion = None
            self.optimizer = None
            self.scaler = None

        self.result = []
        self.best_score = None

    def _load_data(self):
        self.dataset_train = get_dataset(self.cfg.dataset, phase='train')
        self.dataset_eval = get_dataset(self.cfg.dataset, phase='eval')
        self.dataloader_train = get_dataloader(self.dataset_train, self.cfg.dataloader, 'train')
        self.dataloader_eval = get_dataloader(self.dataset_eval, self.cfg.dataloader, 'eval')

    def get_score(self, y_pred, y_true, score='mse'):
        if score == 'mse':
            return np.mean((np.array(y_pred) - np.array(y_true)) ** 2)
        elif score == 'mae':
            return np.mean(np.abs(np.array(y_pred) - np.array(y_true)))
        else:
            raise ValueError(f"Unsupported score type: {score}")

    def is_best_model(self, best_score, current_score):
        if best_score is None:
            return True
        return current_score < best_score

    def _step(self, dataloader: DataLoader, phase: str = 'train', epoch: int = None) -> dict:
        if isinstance(self.model, nn.Module):
            if phase == 'train':
                self.model.train()
            else:
                self.model.eval()

            loss = 0
            y_true = []
            y_pred = []

            num_batches = len(dataloader)

            for data, label in tqdm(dataloader):
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

            loss = loss / num_batches
            score = self.get_score(y_pred, y_true, score=self.score_type)

            return score, loss
        else:
            y_true = []
            y_pred = []

            for data, label in tqdm(dataloader):
                data = data.cpu().numpy()
                label = label.cpu().numpy()
                if phase == 'train':
                    self.model.fit(data, label)
                predictions = self.model.predict(data)
                y_true.extend(list(label))
                y_pred.extend(list(predictions))

            score = self.get_score(y_pred, y_true, score=self.score_type)
            return score

    def train(self):
        self._load_data()
        dl_dct = {"train": self.dataloader_train, "eval": self.dataloader_eval}

        for epoch in range(self.epoch):
            print(f"--------------------------------------")
            print(f"Epoch {epoch + 1}")

            for phase in ["train", "eval"]:
                if isinstance(self.model, nn.Module):
                    score, loss = self._step(dl_dct[phase], phase, epoch)
                    print(f"loss/{phase}:{loss}")
                    print(f"score/{phase}:{score}")
                    self.logger.log_metrics({f"loss/{phase}": loss}, step=epoch)
                    self.logger.log_metrics({f"metrics/{phase}": score}, step=epoch)
                else:
                    score = self._step(dl_dct[phase], phase, epoch)
                    print(f"score/{phase}:{score}")
                    self.logger.log_metrics({f"metrics/{phase}": score}, step=epoch)

                if phase == 'eval' and self.is_best_model(self.best_score, score):
                    print("best model ever!")
                    self.best_score = score
                    self.logger.log_model(model=self.model, model_name="model_best")

        self.logger.log_model(model=self.model, model_name="model_last")

def get_trainer(trainer_cfg=trainer_cfg(), logger=Logger()):
    return MultiModelTrainer(trainer_cfg, logger)




from dataclasses import dataclass, field

@dataclass
class model_cfg:
    name: str = "regression"
    backbone: str = "vgg16"
    pre_train: bool = True
    in_chans: int = 1
    out_dim: int = 3
    n_estimators: int = 100 
    input_size: int = 128  
    hidden_size: int = 64  
    output_size: int = 3  
    custom: dict = field(default_factory=dict)

@dataclass
class dataset_cfg:
    name: str = "custom"
    eval_rate: float = 0.2
    root_dir: str = "/mnt/d/dataset/"
    fold: int = 0
    seed: int = 42
    features: list = field(default_factory=lambda: ["feature1", "feature2", "feature3"])
    target: str = "label"
    property: list = field(default_factory=lambda: ["ZnO", "Ag", "Cu"])
    custom: dict = field(default_factory=lambda: {})


@dataclass
class trainer_cfg:
    name: str = "Trainer"
    seed: int = 42
    epoch: int = 3
    device: str = "cuda:0"
    amp: bool = True
    task: str = "regression"
    optimizer: optimizer_cfg = optimizer_cfg()
    dataset: dataset_cfg = dataset_cfg()
    model: model_cfg = model_cfg()
    dataloader: dataloader_cfg = dataloader_cfg()
    custom: dict = field(default_factory=dict)
