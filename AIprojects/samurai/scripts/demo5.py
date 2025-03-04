import argparse
import os
import os.path as osp
import numpy as np
import cv2
import torch
import gc
import sys
import logging
import psutil
import time
import tempfile
import shutil
from datetime import datetime

sys.path.append("./sam2")
from sam2.build_sam import build_sam2_video_predictor


def setup_logger(log_file=None):
    logger = logging.getLogger('VideoProcessor')
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_memory_usage():
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    gpu_memory = torch.cuda.memory_allocated() / 1024 ** 2 if torch.cuda.is_available() else 0
    return {
        'RAM_Used_MB': memory_info.rss / 1024 ** 2,
        'GPU_Memory_MB': gpu_memory
    }


def load_txt(gt_path, logger):
    logger.info(f"Loading annotations from {gt_path}")
    with open(gt_path, 'r') as f:
        gt = f.readlines()
    prompts = {}
    for fid, line in enumerate(gt):
        x, y, w, h = map(float, line.split(','))
        x, y, w, h = int(x), int(y), int(w), int(h)
        prompts[fid] = ((x, y, x + w, y + h), 0)
    logger.info(f"Loaded {len(prompts)} annotations")
    return prompts


def determine_model_cfg(model_path):
    if "large" in model_path:
        return "configs/sam2.1/sam2.1_hiera_l.yaml"
    elif "base_plus" in model_path:
        return "configs/sam2.1/sam2.1_hiera_b+.yaml"
    elif "small" in model_path:
        return "configs/sam2.1/sam2.1_hiera_s.yaml"
    elif "tiny" in model_path:
        return "configs/sam2.1/sam2.1_hiera_t.yaml"
    else:
        raise ValueError("Unknown model size in path!")


class ChunkedVideoProcessor:
    def __init__(self, video_path, chunk_size=100):
        self.video_path = video_path
        self.chunk_size = chunk_size
        self.cap = cv2.VideoCapture(video_path)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.current_frame = 0

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cap.release()

    def get_next_chunk(self, temp_dir):
        """프레임을 임시 디렉토리에 저장하고 디렉토리 경로 반환"""
        count = 0
        frame_dir = os.path.join(temp_dir, f"chunk_{self.current_frame}")
        os.makedirs(frame_dir, exist_ok=True)

        while count < self.chunk_size and self.current_frame < self.total_frames:
            ret, frame = self.cap.read()
            if not ret:
                break

            # 단순 숫자로 파일 이름 지정
            frame_path = os.path.join(frame_dir, f"{count}.jpg")
            cv2.imwrite(frame_path, frame)

            count += 1
            self.current_frame += 1

        return frame_dir if count > 0 else None


def main(args):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = f"video_processing_{timestamp}.log"
    logger = setup_logger(log_file)

    logger.info("=== Starting Video Processing ===")
    logger.info(f"Arguments: {vars(args)}")

    start_time = time.time()
    chunk_size = 25  # 청크 크기를 25로 유지

    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            logger.info(f"Created temporary directory: {temp_dir}")

            model_cfg = determine_model_cfg(args.model_path)
            logger.info(f"Using model config: {model_cfg}")

            logger.info("Initializing predictor...")
            predictor = build_sam2_video_predictor(model_cfg, args.model_path, device="cuda:0")
            prompts = load_txt(args.txt_path, logger)

            with ChunkedVideoProcessor(args.video_path, chunk_size) as video_processor:
                width = video_processor.width
                height = video_processor.height
                total_frames = video_processor.total_frames

                logger.info(f"Video info - dimensions: {width}x{height}, total frames: {total_frames}")

                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(args.video_output_path, fourcc, 30, (width, height))

                current_frame_idx = 0
                is_first_chunk = True

                while True:
                    chunk_dir = video_processor.get_next_chunk(temp_dir)
                    if chunk_dir is None:
                        break

                    logger.info(f"Processing chunk: frames {current_frame_idx} to {current_frame_idx + chunk_size}")
                    mem_usage = get_memory_usage()
                    logger.info(
                        f"Memory usage - RAM: {mem_usage['RAM_Used_MB']:.1f}MB, GPU: {mem_usage['GPU_Memory_MB']:.1f}MB")

                    try:
                        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
                            state = predictor.init_state(chunk_dir, offload_video_to_cpu=True)

                            if is_first_chunk:
                                bbox, track_label = prompts[0]
                                logger.info(f"Initial bbox: {bbox}, track_label: {track_label}")
                                _, _, masks = predictor.add_new_points_or_box(state, box=bbox, frame_idx=0, obj_id=0)
                                is_first_chunk = False

                            for frame_idx, object_ids, masks in predictor.propagate_in_video(state):
                                if frame_idx % 10 == 0:
                                    mem_usage = get_memory_usage()
                                    logger.info(
                                        f"Processing frame {current_frame_idx + frame_idx}: RAM {mem_usage['RAM_Used_MB']:.1f}MB, GPU {mem_usage['GPU_Memory_MB']:.1f}MB")

                                frame_path = os.path.join(chunk_dir, f"{frame_idx}.jpg")
                                img = cv2.imread(frame_path)

                                for obj_id, mask in zip(object_ids, masks):
                                    mask = mask[0].cpu().numpy()
                                    mask = mask > 0.0

                                    # 바운딩 박스 계산
                                    non_zero_indices = np.argwhere(mask)
                                    if len(non_zero_indices) > 0:
                                        y_min, x_min = non_zero_indices.min(axis=0)
                                        y_max, x_max = non_zero_indices.max(axis=0)

                                        # 마스크 오버레이
                                        mask_img = np.zeros((height, width, 3), np.uint8)
                                        mask_img[mask] = (255, 0, 0)
                                        img = cv2.addWeighted(img, 1, mask_img, 0.2, 0)

                                        # 바운딩 박스 그리기
                                        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)

                                out.write(img)

                    except Exception as e:
                        logger.error(f"Error processing chunk: {str(e)}", exc_info=True)
                        continue
                    finally:
                        # 청크 처리 완료 후 메모리 정리
                        shutil.rmtree(chunk_dir)
                        del state
                        gc.collect()
                        torch.cuda.empty_cache()

                    current_frame_idx += chunk_size

                out.release()
                logger.info("Video saved successfully")

        del predictor
        gc.collect()
        torch.cuda.empty_cache()

        end_time = time.time()
        logger.info(f"Processing completed in {end_time - start_time:.2f} seconds")
        logger.info("=== Processing Finished Successfully ===")

    except Exception as e:
        logger.error(f"Error occurred: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", required=True, help="Input video path or directory of frames.")
    parser.add_argument("--txt_path", required=True, help="Path to ground truth text file.")
    parser.add_argument("--model_path", default="sam2/checkpoints/sam2.1_hiera_base_plus.pt",
                        help="Path to the model checkpoint.")
    parser.add_argument("--video_output_path", default="demo.mp4", help="Path to save the output video.")
    parser.add_argument("--save_to_video", default=True, help="Save results to a video.")
    args = parser.parse_args()
    main(args)