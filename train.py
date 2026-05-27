"""
This script launch a training experiment using the config files.
"""

import pickle
import mlflow
import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt


###############################################################################
#                               Helper Functions                              #
###############################################################################


def confussion_matrix(y_true, y_pred):
    """
    Compute the confusion matrix for a multi-class classification problem.
    """
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax)
    classes = np.arange(cm.shape[0])
    ax.set(
        xticks=classes,
        yticks=classes,
        xlabel="Predicted label",
        ylabel="True label",
        title="Confusion Matrix",
    )
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                cm[i, j],
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )
    fig.tight_layout()
    mlflow.log_figure(fig, "confusion_matrix.png")
    plt.close(fig)


###############################################################################
#                               MAIN FUNCTION                                 #
###############################################################################


@hydra.main(config_path="conf", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    with open(cfg.data.path, "rb") as f:
        # Here, we prepare the data. In the case of Fashion MNIST the only preparation step is to normalize the pixel values to [0, 1].
        data = pickle.load(f)
    X_train, y_train = data["X_train"], data["y_train"]
    X_val, y_val = data["X_val"], data["y_val"]
    X_test, y_test = data["X_test"], data["y_test"]

    X_train, X_val, X_test = X_train / 255.0, X_val / 255.0, X_test / 255.0

    # For this first exemple we use a simple logistic regression model from scikit learn.
    # Here we initialize the model with the hyperparameters specified in the config file: conf/config.yaml
    # https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
    # https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression

    ##TODO##
    # The hyperparameter C is not used in the current implementation, but it can be added to the model initialization if needed.
    # Add it to the config file and to the instantiation of the model !
    model = LogisticRegression(
        solver=cfg.model.solver,
        l1_ratio=cfg.model.l1_ratio,
        max_iter=cfg.training.max_iter,
    )

    # Here we declare the mlflow database for logging the experiment results, the training parameters the model hyperparameters
    # and the metrics.
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    # This line sets the experiment name in mlflow,
    # it will create a new experiment, if it doesn't exist or log to the existing one if it does.
    mlflow.set_experiment(cfg.exp_name)
    with mlflow.start_run():
        mlflow.set_tag("exp_name", cfg.exp_name)
        # Log config yaml file as an artifact
        mlflow.log_text(OmegaConf.to_yaml(cfg), "config_exp.yaml")
        # Log training parameters from the config file
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
        # Log the confusion matrix for the test set
        y_test_pred = model.predict(X_test)
        confussion_matrix(y_test, y_test_pred)


if __name__ == "__main__":
    main()
