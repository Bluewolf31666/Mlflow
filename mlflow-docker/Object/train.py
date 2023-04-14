# The data set used in this example is from http://archive.ics.uci.edu/ml/datasets/Wine+Quality
# P. Cortez, A. Cerdeira, F. Almeida, T. Matos and J. Reis.
# Modeling wine preferences by data mining from physicochemical properties. In Decision Support Systems, Elsevier, 47(4):547-553, 2009.
import os
import sys
import click
import warnings

import pandas as pd
import numpy as np
from itertools import cycle
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import lasso_path, enet_path
from urllib.parse import urlparse
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature

import logging

from dotenv import load_dotenv
import os

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

# MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "")


@click.command()
@click.option(
    "--experiment-name", help="Experiment name", default="sklearn-test", type=str
)
@click.option("--model-name", help="Registered model name", default=None, type=str)
@click.option("--data-path", help="Data path", default=None, type=str)
@click.option(
    "--auto-log",
    help="Explicitly log params, metrics and model with mlflow.autolog",
    default=False,
    type=bool,
)
@click.option(
    "--custom-log",
    help="Explicitly log params, metrics and model with mlflow.log_",
    default=False,
    type=bool,
)
#####
def main(
    experiment_name,
    model_name,
    data_path,
    auto_log,
    custom_log,
):
    # 실험 생성 & 실험 id 가져오기
    experiment = mlflow.set_experiment("scikitLearn-model")
    experiment_id = experiment.experiment_id

    auto_log = "" if not auto_log or auto_log == "None" else auto_log
    custom_log = "" if not custom_log or custom_log == "None" else custom_log

    if not auto_log and not custom_log:
        auto_log = True

    print("Log method:")
    if auto_log:
        print("autolog")
        print("=======================\n")
        mlflow.sklearn.autolog()

    print("Options:")
    for k, v in locals().items():
        print(f"  {k}: {v}")

    with mlflow.start_run(experiment_id=experiment_id, run_name=experiment_name) as run:
        print("MLflow:")
        print("  run_id:", run.info.run_id)
        print("  experiment_id:", run.info.experiment_id)
        print("  experiment_run_name:", run.info.run_name)
        print("=======================\n")

        mlflow.set_tag("version.mlflow", mlflow.__version__)
        # mlflow.set_tag("version.mlflow.sklearn", sklearn.__version__)

        train(
            run,
            experiment_name,
            model_name,
            data_path,
            auto_log,
            custom_log,
        )


def train(
    run,
    experiment_name,
    model_name,
    data_path,
    auto_log,
    custom_log,
):

    print("Loading data...")
    # Read the wine-quality csv file from the URL
    csv_url = "https://raw.githubusercontent.com/mlflow/mlflow/master/tests/datasets/winequality-red.csv"
    try:
        data = pd.read_csv(csv_url, sep=";")
    except Exception as e:
        logger.exception(
            "Unable to download training & test CSV, check your internet connection. Error: %s",
            e,
        )

    # Split the data into training and test sets. (0.75, 0.25) split.
    train, test = train_test_split(data)

    # The predicted column is "quality" which is a scalar from [3, 9]
    train_x = train.drop(["quality"], axis=1)
    test_x = test.drop(["quality"], axis=1)
    train_y = train[["quality"]]
    test_y = test[["quality"]]

    print(len(train_x), "train sequences")
    print(len(test_x), "test sequences")
    print("=======================\n")

    alpha = 0.5
    l1_ratio = 0.5

    lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
    lr.fit(train_x, train_y)

    predicted_qualities = lr.predict(test_x)

    (rmse, mae, r2) = _eval_metrics(test_y, predicted_qualities)

    print("Elasticnet model (alpha={:f}, l1_ratio={:f}):".format(
        alpha, l1_ratio))
    # print("  RMSE: %s" % rmse)
    # print("  MAE: %s" % mae)
    # print("  R2: %s" % r2)

    if custom_log:
        mlflow.log_param("alpha", alpha)
        mlflow.log_param("l1_ratio", l1_ratio)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("mae", mae)

    # input / output data log in model schema
    signature = infer_signature(train_x, predicted_qualities)

    mlflow.sklearn.log_model(
        lr,
        "model",
        # registered_model_name=model_name,  # Model Registry
        signature=signature,
    )


def _eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2


if __name__ == "__main__":
    main()
