import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional


class PCMReader:
    def __init__(self, sample_rate: int = 16000, bit_depth: int = 16):
        """
        PCM 파일을 읽고 처리하는 클래스
        Args:
            sample_rate: 샘플링 레이트 (기본값: 16kHz)
            bit_depth: 비트 깊이 (기본값: 16bit)
        """
        self.sample_rate = sample_rate
        self.bit_depth = bit_depth
        self.max_value = 2 ** (bit_depth - 1)

    def read_pcm(self, file_path: str) -> Tuple[np.ndarray, float]:
        """
        PCM 파일을 읽어서 numpy 배열로 반환
        Args:
            file_path: PCM 파일 경로
        Returns:
            audio_data: 오디오 데이터 배열
            duration: 오디오 길이(초)
        """
        with open(file_path, 'rb') as f:
            audio_data = np.frombuffer(f.read(), dtype=np.int16)

        # 정규화 (-1 ~ 1 범위로)
        audio_data = audio_data.astype(np.float32) / self.max_value
        duration = len(audio_data) / self.sample_rate

        return audio_data, duration

    def get_audio_info(self, audio_data: np.ndarray) -> dict:
        """
        오디오 데이터의 기본 정보를 반환
        """
        return {
            'samples': len(audio_data),
            'duration': len(audio_data) / self.sample_rate,
            'max_amplitude': np.max(np.abs(audio_data)),
            'rms': np.sqrt(np.mean(np.square(audio_data))),
        }

    def plot_waveform(self, audio_data: np.ndarray, title: Optional[str] = None) -> None:
        """
        오디오 파형을 시각화
        """
        duration = len(audio_data) / self.sample_rate
        time = np.linspace(0, duration, len(audio_data))

        plt.figure(figsize=(12, 4))
        plt.plot(time, audio_data)
        plt.title(title or 'Waveform')
        plt.xlabel('Time (seconds)')
        plt.ylabel('Amplitude')
        plt.grid(True)
        plt.show()


def main():
    # 사용 예시
    reader = PCMReader()
    data_dir = './data'

    # 첫 번째 PCM 파일 찾기
    pcm_files = [f for f in os.listdir(data_dir) if f.endswith('.pcm')]
    if not pcm_files:
        print("PCM 파일을 찾을 수 없습니다.")
        return

    # 첫 번째 파일로 테스트
    test_file = os.path.join(data_dir, pcm_files[0])
    print(f"테스트 파일: {test_file}")

    # 파일 읽기 및 정보 출력
    audio_data, duration = reader.read_pcm(test_file)
    info = reader.get_audio_info(audio_data)

    print("\n오디오 정보:")
    print(f"샘플 수: {info['samples']:,}")
    print(f"재생 시간: {info['duration']:.2f}초")
    print(f"최대 진폭: {info['max_amplitude']:.4f}")
    print(f"RMS 값: {info['rms']:.4f}")

    # 파형 시각화
    reader.plot_waveform(audio_data, f'Waveform - {os.path.basename(test_file)}')


if __name__ == "__main__":
    main()