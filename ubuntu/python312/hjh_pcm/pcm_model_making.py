import os
import numpy as np
from tqdm import tqdm
from typing import List, Tuple, Dict
from dataclasses import dataclass
from collections import defaultdict


@dataclass
class AudioStats:
    duration: float
    max_amplitude: float
    rms: float


class DatasetAnalyzer:
    def __init__(self, data_dir: str, sample_rate: int = 16000):
        self.data_dir = data_dir
        self.sample_rate = sample_rate
        self.audio_stats = {}
        self.text_stats = {}

    def get_file_pairs(self) -> List[Tuple[str, str]]:
        """PCM과 텍스트 파일 쌍을 재귀적으로 찾아 반환"""
        pairs = []

        for root, _, files in os.walk(self.data_dir):
            pcm_files = {f for f in files if f.endswith('.pcm')}
            txt_files = {f.replace('.pcm', '.txt') for f in pcm_files}

            for pcm in pcm_files:
                txt = pcm.replace('.pcm', '.txt')
                if txt in txt_files:
                    # 전체 경로에서 data_dir을 제외한 상대 경로를 저장
                    rel_path = os.path.relpath(root, self.data_dir)
                    if rel_path == '.':
                        pairs.append((pcm, txt))
                    else:
                        pairs.append((os.path.join(rel_path, pcm),
                                      os.path.join(rel_path, txt)))

        return sorted(pairs)

    def analyze_audio(self, pcm_path: str) -> AudioStats:
        """PCM 파일 분석"""
        with open(pcm_path, 'rb') as f:
            audio_data = np.frombuffer(f.read(), dtype=np.int16)

        audio_data = audio_data.astype(np.float32) / 32768.0
        duration = len(audio_data) / self.sample_rate
        max_amplitude = np.max(np.abs(audio_data))
        rms = np.sqrt(np.mean(np.square(audio_data)))

        return AudioStats(duration, max_amplitude, rms)

    def analyze_text(self, text_path: str) -> Dict:
        """텍스트 파일 분석"""
        try:
            # CP949(EUC-KR) 인코딩으로 먼저 시도
            with open(text_path, 'r', encoding='cp949') as f:
                text = f.read().strip()
        except UnicodeDecodeError:
            try:
                # 실패하면 UTF-8로 시도
                with open(text_path, 'r', encoding='utf-8') as f:
                    text = f.read().strip()
            except UnicodeDecodeError:
                print(f"Warning: {text_path} 파일을 읽는데 실패했습니다. 빈 텍스트로 대체합니다.")
                text = ""

        return {
            'char_count': len(text),
            'word_count': len(text.split()),
            'text': text
        }

    def analyze_dataset(self) -> Dict:
        """전체 데이터셋 분석"""
        pairs = self.get_file_pairs()
        print(f"총 {len(pairs)}개의 파일 쌍을 찾았습니다.")

        durations = []
        max_amplitudes = []
        rms_values = []
        char_counts = []
        word_counts = []

        for pcm, txt in tqdm(pairs, desc="데이터셋 분석 중"):
            pcm_path = os.path.join(self.data_dir, pcm)
            txt_path = os.path.join(self.data_dir, txt)

            if not os.path.exists(pcm_path) or not os.path.exists(txt_path):
                print(f"Warning: {pcm} 또는 {txt} 파일이 없습니다. 건너뜁니다.")
                continue

            # 오디오 분석
            audio_stats = self.analyze_audio(pcm_path)
            durations.append(audio_stats.duration)
            max_amplitudes.append(audio_stats.max_amplitude)
            rms_values.append(audio_stats.rms)

            # 텍스트 분석
            text_stats = self.analyze_text(txt_path)
            char_counts.append(text_stats['char_count'])
            word_counts.append(text_stats['word_count'])

        return {
            'audio': {
                'total_duration': sum(durations),
                'avg_duration': np.mean(durations),
                'std_duration': np.std(durations),
                'min_duration': min(durations),
                'max_duration': max(durations),
                'avg_max_amplitude': np.mean(max_amplitudes),
                'avg_rms': np.mean(rms_values)
            },
            'text': {
                'total_chars': sum(char_counts),
                'avg_chars': np.mean(char_counts),
                'total_words': sum(word_counts),
                'avg_words': np.mean(word_counts)
            }
        }


def main():
    data_dir = 'D:/korean/KsponSpeech_01'
    analyzer = DatasetAnalyzer(data_dir)
    stats = analyzer.analyze_dataset()

    print("\n=== 데이터셋 통계 ===")
    print(f"\n[오디오 통계]")
    print(f"총 재생 시간: {stats['audio']['total_duration']:.2f}초")
    print(f"평균 재생 시간: {stats['audio']['avg_duration']:.2f}초")
    print(f"재생 시간 표준편차: {stats['audio']['std_duration']:.2f}초")
    print(f"최소 재생 시간: {stats['audio']['min_duration']:.2f}초")
    print(f"최대 재생 시간: {stats['audio']['max_duration']:.2f}초")
    print(f"평균 최대 진폭: {stats['audio']['avg_max_amplitude']:.4f}")
    print(f"평균 RMS: {stats['audio']['avg_rms']:.4f}")

    print(f"\n[텍스트 통계]")
    print(f"총 문자 수: {stats['text']['total_chars']:,}")
    print(f"평균 문자 수: {stats['text']['avg_chars']:.1f}")
    print(f"총 단어 수: {stats['text']['total_words']:,}")
    print(f"평균 단어 수: {stats['text']['avg_words']:.1f}")


if __name__ == "__main__":
    main()