import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pickle
import mlflow
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from mlflow.models.signature import infer_signature
from torch.utils.data import DataLoader, TensorDataset
import lightning.pytorch as pl
from lightning.pytorch import Trainer
from lightning.pytorch.loggers import MLFlowLogger
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
import torchmetrics
import optuna
import os


def load_data(file_path: str = 'Churn_Modelling.csv') -> pd.DataFrame:
    df = pd.read_csv(file_path)
    df.drop(columns=['RowNumber', 'CustomerId', 'Surname'], inplace=True)
    return df


def transform_data(df: pd.DataFrame) -> tuple[pd.DataFrame, LabelEncoder, OneHotEncoder]:
    label_encoder = LabelEncoder()
    df['Gender'] = label_encoder.fit_transform(df['Gender'])

    one_hot = OneHotEncoder()
    geo_encoded = one_hot.fit_transform(df[['Geography']]).toarray()
    geo_df = pd.DataFrame(geo_encoded, columns=one_hot.get_feature_names_out(['Geography']))
    df = pd.concat([df.reset_index(drop=True), geo_df], axis=1)
    df.drop(columns=['Geography'], inplace=True)

    return df, label_encoder, one_hot


def dump_pickle(obj, file_path: str) -> None:
    with open(file_path, 'wb') as file:
        pickle.dump(obj, file)


def load_pickle(file_path: str) -> object:
    with open(file_path, 'rb') as file:
        return pickle.load(file)


def split_dataset(df: pd.DataFrame, test_size: float = 0.2, val_size: float = 0.1, random_state: int = 42):
    X = df.drop(columns=['Exited'])
    y = df['Exited']
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=val_size, random_state=random_state, stratify=y_trainval
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def scale_dataset(X_train, X_val, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_val_scaled, X_test_scaled, scaler


class ChurnModel(pl.LightningModule):
    def __init__(self, input_dim: int, output_dim: int = 1):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, output_dim)

        self.train_acc = torchmetrics.Accuracy(task='binary')
        self.train_precision = torchmetrics.Precision(task='binary')
        self.train_recall = torchmetrics.Recall(task='binary')
        self.train_f1 = torchmetrics.F1Score(task='binary')
        self.train_auroc = torchmetrics.AUROC(task='binary')

        self.val_acc = torchmetrics.Accuracy(task='binary')
        self.test_acc = torchmetrics.Accuracy(task='binary')

        self.lr = 0.001

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x).squeeze()
        preds = (y_hat > 0.5).int()
        loss = F.binary_cross_entropy(y_hat, y.float())

        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_acc', self.train_acc(preds, y.int()), on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_precision', self.train_precision(preds, y.int()), on_step=True, on_epoch=True)
        self.log('train_recall', self.train_recall(preds, y.int()), on_step=True, on_epoch=True)
        self.log('train_f1', self.train_f1(preds, y.int()), on_step=True, on_epoch=True)
        self.log('train_auroc', self.train_auroc(y_hat, y.int()), on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x).squeeze()
        preds = (y_hat > 0.5).int()
        loss = F.binary_cross_entropy(y_hat, y.float())
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', self.val_acc(preds, y.int()), prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x).squeeze()
        preds = (y_hat > 0.5).int()
        loss = F.binary_cross_entropy(y_hat, y.float())
        self.log('test_loss', loss, prog_bar=True)
        self.log('test_acc', self.test_acc(preds, y.int()), prog_bar=True)
        return loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr)


def objective(trial):
    hidden1 = trial.suggest_categorical("hidden1", [32, 64, 128])
    hidden2 = trial.suggest_categorical("hidden2", [16, 32, 64])
    learning_rate = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])

    df = load_data("Churn_Modelling.csv")
    df, label_encoder, one_hot = transform_data(df)
    dump_pickle(label_encoder, "label_encoder.pkl")
    dump_pickle(one_hot, "one_hot_encoder.pkl")

    X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(df)
    X_train_scaled, X_val_scaled, X_test_scaled, scaler = scale_dataset(X_train, X_val, X_test)
    dump_pickle(scaler, "scaler.pkl")

    train_dataset = TensorDataset(torch.tensor(X_train_scaled).float(), torch.tensor(y_train.values).float())
    val_dataset = TensorDataset(torch.tensor(X_val_scaled).float(), torch.tensor(y_val.values).float())
    test_dataset = TensorDataset(torch.tensor(X_test_scaled).float(), torch.tensor(y_test.values).float())

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    model = ChurnModel(input_dim=X_train_scaled.shape[1])
    model.fc1 = nn.Linear(X_train_scaled.shape[1], hidden1)
    model.fc2 = nn.Linear(hidden1, hidden2)
    model.fc3 = nn.Linear(hidden2, 1)
    model.lr = learning_rate

    mlflow.set_tracking_uri("http://localhost:5001")
    mlflow.set_experiment("Churn Prediction Optuna")

    with mlflow.start_run() as run:
        mlflow_logger = MLFlowLogger(tracking_uri="http://localhost:5001", run_id=run.info.run_id)
        mlflow.log_params({
            "hidden1": hidden1,
            "hidden2": hidden2,
            "learning_rate": learning_rate,
            "batch_size": batch_size
        })

        checkpoint_callback = ModelCheckpoint(monitor='val_loss', dirpath='checkpoints', filename='optuna-best-{epoch:02d}-{val_loss:.2f}', save_top_k=1, mode='min')
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, mode='min')

        trainer = Trainer(
            max_epochs=15,
            logger=mlflow_logger,
            callbacks=[checkpoint_callback, early_stopping],
            accelerator='auto',
            devices=1,
            enable_progress_bar=True
        )

        trainer.fit(model, train_loader, val_loader)
        val_loss = trainer.callback_metrics["val_loss"]
        trainer.test(model, test_loader)

        sample_input = torch.tensor(X_test_scaled[:5]).float()
        onnx_path = "churn_model_optuna.onnx"
        torch.onnx.export(
            model,
            sample_input,
            onnx_path,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
            opset_version=11
        )
        mlflow.log_artifact(onnx_path)
        mlflow.log_artifact("label_encoder.pkl")
        mlflow.log_artifact("one_hot_encoder.pkl")
        mlflow.log_artifact("scaler.pkl")
        mlflow.log_artifact("Churn_Modelling.csv")

        return val_loss


if __name__ == "__main__":
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=15)

    print("Best trial:")
    print("  Value: ", study.best_value)
    print("  Params: ", study.best_params)

    best_params = study.best_params
    hidden1 = best_params["hidden1"]
    hidden2 = best_params["hidden2"]
    learning_rate = best_params["lr"]
    batch_size = best_params["batch_size"]

    df = load_data("Churn_Modelling.csv")
    df, label_encoder, one_hot = transform_data(df)
    dump_pickle(label_encoder, "label_encoder.pkl")
    dump_pickle(one_hot, "one_hot_encoder.pkl")

    X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(df)
    X_train_scaled, X_val_scaled, X_test_scaled, scaler = scale_dataset(X_train, X_val, X_test)
    dump_pickle(scaler, "scaler.pkl")

    train_dataset = TensorDataset(torch.tensor(X_train_scaled).float(), torch.tensor(y_train.values).float())
    val_dataset = TensorDataset(torch.tensor(X_val_scaled).float(), torch.tensor(y_val.values).float())
    test_dataset = TensorDataset(torch.tensor(X_test_scaled).float(), torch.tensor(y_test.values).float())

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    model = ChurnModel(input_dim=X_train_scaled.shape[1])
    model.fc1 = nn.Linear(X_train_scaled.shape[1], hidden1)
    model.fc2 = nn.Linear(hidden1, hidden2)
    model.fc3 = nn.Linear(hidden2, 1)
    model.lr = learning_rate

    mlflow.set_tracking_uri("http://localhost:5001")
    mlflow.set_experiment("Churn Prediction Optuna")

    with mlflow.start_run(run_name="best_model"):
        mlflow.log_params(best_params)

        trainer = Trainer(
            max_epochs=30,
            logger=MLFlowLogger(tracking_uri="http://localhost:5001", run_name="best_model"),
            callbacks=[
                EarlyStopping(monitor='val_loss', patience=5, mode='min')
            ],
            accelerator="auto",
            devices=1,
            enable_progress_bar=True
        )

        trainer.fit(model, train_loader, val_loader)
        trainer.test(model, test_loader)

        sample_input = torch.tensor(X_test_scaled[:5]).float()
        onnx_path = "churn_model_best.onnx"
        torch.onnx.export(
            model,
            sample_input,
            onnx_path,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
            opset_version=11
        )
        mlflow.log_artifact(onnx_path)
        mlflow.log_artifact("label_encoder.pkl")
        mlflow.log_artifact("one_hot_encoder.pkl")
        mlflow.log_artifact("scaler.pkl")
        mlflow.log_artifact("Churn_Modelling.csv")

        model_uri = f"runs:/{mlflow.active_run().info.run_id}/{onnx_path}"
        mlflow.register_model(
            model_uri=model_uri,
            name="ChurnPredictionONNX"
        )

        print("Best model retrained and logged to MLflow as ONNX.")