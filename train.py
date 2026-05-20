"""
This script launch a training experiment using the config files.
"""
import pickle
import mlflow
import hydra
from omegaconf import DictConfig, OmegaConf
from sklearn.linear_model import LogisticRegression

@hydra.main(config_path='conf',
            config_name='config',
            version_base='1.3')
def main(cfg: DictConfig):

    with open(cfg.data.path, 'rb') as f:
        data = pickle.load(f)
        X_train, y_train = data['X_train'], data['y_train']
        X_val, y_val = data['X_val'], data['y_val']
        X_test, y_test = data['X_test'], data['y_test']

        X_train, X_val, X_test = X_train / 255.0, X_val / 255.0, X_test / 255.0
        model = LogisticRegression(solver=cfg.model.solver,
                                   l1_ratio=cfg.model.l1_ratio, 
                                   C=cfg.model.C,
                                   max_iter=cfg.training.max_iter)



    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment(cfg.exp_name)
    with mlflow.start_run():
        mlflow.set_tag("exp_name", cfg.exp_name)
        # Log config as an artifact
        mlflow.log_text(OmegaConf.to_yaml(cfg), "config_exp.yaml")
        # Log training parameters
        for key, value in cfg.training.items():
            mlflow.log_param(key, value)
        # Log model hyperparameters
        for key, value in cfg.model.items():
            mlflow.log_param(key, value)
        # Train the model
        model.fit(X_train, y_train)
        # Log train/val/test metrics
        train_accuracy = model.score(X_train, y_train)
        mlflow.log_metric("train/accuracy", float(train_accuracy))
        val_accuracy = model.score(X_val, y_val)
        mlflow.log_metric("val/accuracy", float(val_accuracy))
        test_accuracy = model.score(X_test, y_test)
        mlflow.log_metric("test/accuracy", float(test_accuracy))

if __name__ == "__main__":
    main()