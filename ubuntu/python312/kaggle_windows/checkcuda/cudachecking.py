#pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

import subprocess
import sys
import os
import logging
from datetime import datetime


# 로깅 설정
def setup_logging():
    log_dir = "cuda_logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    log_file = os.path.join(log_dir, f"cuda_check_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)


def check_cuda_installation():
    logger = setup_logging()

    try:
        # Python 환경 정보
        logger.info("=== Python 환경 정보 ===")
        logger.info(f"Python 버전: {sys.version}")
        logger.info(f"Python 경로: {sys.executable}")

        # CUDA 관련 환경 변수 확인
        logger.info("\n=== CUDA 환경 변수 ===")
        cuda_path = os.environ.get('CUDA_PATH')
        logger.info(f"CUDA_PATH: {cuda_path}")

        # nvidia-smi 실행
        logger.info("\n=== NVIDIA-SMI 정보 ===")
        try:
            nvidia_smi = subprocess.check_output(['nvidia-smi'], text=True)
            logger.info(f"nvidia-smi 출력:\n{nvidia_smi}")
        except subprocess.CalledProcessError as e:
            logger.error("nvidia-smi 실행 실패. NVIDIA 드라이버가 설치되어 있지 않을 수 있습니다.")
            logger.error(f"에러: {str(e)}")

        # PyTorch CUDA 확인
        logger.info("\n=== PyTorch CUDA 지원 확인 ===")
        try:
            import torch
            logger.info(f"PyTorch 버전: {torch.__version__}")
            logger.info(f"CUDA 사용 가능: {torch.cuda.is_available()}")
            if torch.cuda.is_available():
                logger.info(f"현재 CUDA 버전: {torch.version.cuda}")
                logger.info(f"GPU 개수: {torch.cuda.device_count()}")
                for i in range(torch.cuda.device_count()):
                    logger.info(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        except ImportError:
            logger.warning("PyTorch가 설치되어 있지 않습니다.")

        # CUDA 툴킷 버전 확인
        logger.info("\n=== CUDA 툴킷 버전 확인 ===")
        try:
            nvcc_output = subprocess.check_output(['nvcc', '--version'], text=True)
            logger.info(f"NVCC 버전 정보:\n{nvcc_output}")
        except subprocess.CalledProcessError as e:
            logger.error("NVCC 실행 실패. CUDA 툴킷이 설치되어 있지 않을 수 있습니다.")
            logger.error(f"에러: {str(e)}")
        except FileNotFoundError:
            logger.error("NVCC를 찾을 수 없습니다. CUDA 툴킷이 설치되어 있지 않거나 PATH에 등록되지 않았습니다.")

    except Exception as e:
        logger.error(f"체크 중 예외 발생: {str(e)}", exc_info=True)

    logger.info("\n=== 체크 완료 ===")


if __name__ == "__main__":
    check_cuda_installation()