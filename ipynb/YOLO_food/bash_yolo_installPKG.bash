#!/bin/bash

echo "YOLO 환경 설정을 시작합니다..."

# Python 버전 확인
echo "Python 버전 확인 중..."
python3 --version

# pip 업그레이드
echo "pip 업그레이드 중..."
python3 -m pip install --upgrade pip

# 기본 패키지들 설치
echo "기본 패키지들 설치 중..."
python3 -m pip install numpy
python3 -m pip install opencv-python
python3 -m pip install Pillow
python3 -m pip install matplotlib
python3 -m pip install pyyaml
python3 -m pip install requests
python3 -m pip install tqdm

# PyTorch 설치 (CPU 버전)
echo "PyTorch 설치 중..."
python3 -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# GPU가 있는 경우 아래 명령어를 사용하세요 (위 명령어 대신):
# python3 -m pip install torch torchvision torchaudio

# Ultralytics YOLO 설치
echo "Ultralytics YOLO 설치 중..."
python3 -m pip install ultralytics

# 추가 유용한 패키지들
echo "추가 패키지들 설치 중..."
python3 -m pip install seaborn
python3 -m pip install pandas
python3 -m pip install scipy

# 설치 확인
echo "설치 확인 중..."
python3 -c "import ultralytics; print('Ultralytics YOLO 설치 성공!')"
python3 -c "from ultralytics import YOLO; print('YOLO 모듈 import 성공!')"

echo "YOLO 환경 설정이 완료되었습니다!"
echo "이제 06trainfood.py를 실행할 수 있습니다."
