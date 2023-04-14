import boto3
import os
import sys
import mlflow
import pandas as pd


from sklearn.model_selection import train_test_split
from mlflow.tracking import MlflowClient

from dotenv import load_dotenv

import logging

load_dotenv()


TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "")
logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)


client = MlflowClient(tracking_uri=TRACKING_URI)

csv_url = "https://raw.githubusercontent.com/mlflow/mlflow/master/tests/datasets/winequality-red.csv"
try:
    data = pd.read_csv(csv_url, sep=";")
except Exception as e:
    logger.exception(
        "Unable to download training & test CSV, check your internet connection. Error: %s",
        e,
    )

_, test = train_test_split(data)
data = test.drop(["quality"], axis=1)

# 모델 repository 검색 및 조회
model_name = "WineQualityModel"
filter_string = "name='{}'".format(model_name)
results = client.search_model_versions(filter_string)

print("\n==========")
print("Search model registry info.")
for res in results:
    print(
        "name={}; run_id={}; version={}; current_stage={}".format(
            res.name, res.run_id, res.version, res.current_stage
        )
    )

# Production stage 모델 버전 선택
deploy_version = None
for res in results:
    if res.current_stage == "Production":
        deploy_version = res.version


if deploy_version == None:
    print("\n==========")
    print("Unknown deploy version")
    sys.exit()
else:
    print("\n==========")
    print("deploy version:", deploy_version)

model_uri = client.get_model_version_download_uri(model_name, deploy_version)
model = mlflow.pyfunc.load_model(model_uri)

predictions = model.predict(data)
print("\n==========")
print(predictions)
