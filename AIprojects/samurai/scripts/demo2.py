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
from datetime import datetime

sys.path.append("./sam2")
from sam2.build_sam import build_sam2_video_predictor


# 로깅 설정
def setup_logger(log_file=None):
    logger = logging.getLogger('VideoProcessor')
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    # 콘솔 핸들러
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # 파일 핸들러 (지정된 경우)
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


color = [(255, 0, 0)]


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


def prepare_frames_or_path(video_path, logger):
    logger.info(f"Preparing input from: {video_path}")
    if video_path.endswith(".mp4") or osp.isdir(video_path):
        return video_path
    else:
        raise ValueError("Invalid video_path format. Should be .mp4 or a directory of jpg frames.")


def main(args):
    # 로그 파일 이름 설정 (현재 시간 포함)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = f"video_processing_{timestamp}.log"
    logger = setup_logger(log_file)

    logger.info("=== Starting Video Processing ===")
    logger.info(f"Arguments: {vars(args)}")

    start_time = time.time()

    try:
        model_cfg = determine_model_cfg(args.model_path)
        logger.info(f"Using model config: {model_cfg}")

        logger.info("Initializing predictor...")
        predictor = build_sam2_video_predictor(model_cfg, args.model_path, device="cuda:0")
        frames_or_path = prepare_frames_or_path(args.video_path, logger)
        prompts = load_txt(args.txt_path, logger)

        if args.save_to_video:
            if osp.isdir(args.video_path):
                logger.info("Loading frames from directory...")
                frames = sorted(
                    [osp.join(args.video_path, f) for f in os.listdir(args.video_path) if f.endswith(".jpg")])
                loaded_frames = [cv2.imread(frame_path) for frame_path in frames]
                height, width = loaded_frames[0].shape[:2]
                logger.info(f"Loaded {len(loaded_frames)} frames, dimensions: {width}x{height}")
            else:
                logger.info("Loading frames from video file...")
                cap = cv2.VideoCapture(args.video_path)
                loaded_frames = []
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    loaded_frames.append(frame)
                cap.release()
                height, width = loaded_frames[0].shape[:2]
                logger.info(f"Loaded {len(loaded_frames)} frames, dimensions: {width}x{height}")

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(args.video_output_path, fourcc, 30, (width, height))
        logger.info(f"Initialized video writer: {args.video_output_path}")

        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
            logger.info("Initializing state...")
            state = predictor.init_state(frames_or_path, offload_video_to_cpu=True)

            bbox, track_label = prompts[0]
            logger.info(f"Initial bbox: {bbox}, track_label: {track_label}")

            _, _, masks = predictor.add_new_points_or_box(state, box=bbox, frame_idx=0, obj_id=0)

            logger.info("Starting frame processing...")
            frame_count = 0
            for frame_idx, object_ids, masks in predictor.propagate_in_video(state):
                frame_count += 1
                if frame_count % 10 == 0:  # 10프레임마다 로그 출력
                    mem_usage = get_memory_usage()
                    logger.info(
                        f"Processing frame {frame_count}: RAM {mem_usage['RAM_Used_MB']:.1f}MB, GPU {mem_usage['GPU_Memory_MB']:.1f}MB")

                mask_to_vis = {}
                bbox_to_vis = {}

                for obj_id, mask in zip(object_ids, masks):
                    mask = mask[0].cpu().numpy()
                    mask = mask > 0.0
                    non_zero_indices = np.argwhere(mask)
                    if len(non_zero_indices) == 0:
                        bbox = [0, 0, 0, 0]
                    else:
                        y_min, x_min = non_zero_indices.min(axis=0).tolist()
                        y_max, x_max = non_zero_indices.max(axis=0).tolist()
                        bbox = [x_min, y_min, x_max - x_min, y_max - y_min]
                    bbox_to_vis[obj_id] = bbox
                    mask_to_vis[obj_id] = mask

                if args.save_to_video:
                    img = loaded_frames[frame_idx]
                    for obj_id, mask in mask_to_vis.items():
                        mask_img = np.zeros((height, width, 3), np.uint8)
                        mask_img[mask] = color[(obj_id + 1) % len(color)]
                        img = cv2.addWeighted(img, 1, mask_img, 0.2, 0)

                    for obj_id, bbox in bbox_to_vis.items():
                        cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]),
                                      color[obj_id % len(color)], 2)

                    out.write(img)

            logger.info(f"Processed total {frame_count} frames")
            if args.save_to_video:
                out.release()
                logger.info("Video saved successfully")

        del predictor, state
        gc.collect()
        torch.clear_autocast_cache()
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