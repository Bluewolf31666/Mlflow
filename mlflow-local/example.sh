 $ export MLFLOW_TRACKING_URI="http://localhost:5000"

# mlflow ui 를 수행한 디렉토리와 같은 디렉토리에 train_diabetes.py 파일이 있는 경로로 이동
cd /home/wsl/Mlflow/mlflow-local

# example 코드를 실행 후 mlflow 에 기록되는 것 확인
python train_diabetes.py

# 다양한 parameter 로 테스트 후 mlflow 실험 기록 확인
python train_diabetes.py  0.01 0.01
python train_diabetes.py  0.01 0.75
python train_diabetes.py  0.01 1.0
python train_diabetes.py  0.05 1.0
python train_diabetes.py  0.05 0.01
python train_diabetes.py  0.5 0.8
python train_diabetes.py  0.8 1.0
