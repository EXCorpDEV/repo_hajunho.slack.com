import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
import numpy as np
from typing import List, Tuple, Dict
from tqdm import tqdm
import random
from torch.nn.utils.rnn import pad_sequence


class SubsetKsponDataset(Dataset):
    def __init__(self, data_dir: str, subset_size: int = 1000):
        self.data_dir = data_dir
        self.subset_size = subset_size
        self.file_pairs = self._get_subset_pairs()
        self.char_to_index = self._create_vocab()
        print(f"총 클래스 수 (문자 종류): {len(self.char_to_index)}")

    def _get_subset_pairs(self) -> List[Tuple[str, str]]:
        """전체 데이터에서 일부만 랜덤 선택"""
        all_pairs = []
        for root, _, files in os.walk(self.data_dir):
            pcm_files = {f for f in files if f.endswith('.pcm')}
            txt_files = {f.replace('.pcm', '.txt') for f in pcm_files}

            for pcm in pcm_files:
                txt = pcm.replace('.pcm', '.txt')
                if txt in txt_files:
                    rel_path = os.path.relpath(root, self.data_dir)
                    if rel_path == '.':
                        all_pairs.append((pcm, txt))
                    else:
                        all_pairs.append((os.path.join(rel_path, pcm),
                                          os.path.join(rel_path, txt)))

        # 랜덤 선택
        subset_pairs = random.sample(all_pairs, min(self.subset_size, len(all_pairs)))
        print(f"선택된 데이터 수: {len(subset_pairs)}")
        return sorted(subset_pairs)

    def _create_vocab(self) -> Dict[str, int]:
        """선택된 subset에서만 vocabulary 생성"""
        chars = {'<pad>', '<sos>', '<eos>'}  # 특수 토큰 추가

        for _, txt_file in self.file_pairs:
            try:
                with open(os.path.join(self.data_dir, txt_file), 'r', encoding='cp949') as f:
                    text = f.read().strip()
                    chars.update(list(text))
            except UnicodeDecodeError:
                print(f"Warning: {txt_file} 파일을 읽는데 실패했습니다.")

        return {char: idx for idx, char in enumerate(sorted(chars))}

    def __len__(self) -> int:
        return len(self.file_pairs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int, int]:
        pcm_path = os.path.join(self.data_dir, self.file_pairs[idx][0])
        txt_path = os.path.join(self.data_dir, self.file_pairs[idx][1])

        # PCM 파일 로드
        with open(pcm_path, 'rb') as f:
            audio_data = np.frombuffer(f.read(), dtype=np.int16)
        waveform = torch.FloatTensor(audio_data) / 32768.0

        # 텍스트 파일 로드
        with open(txt_path, 'r', encoding='cp949') as f:
            text = f.read().strip()

        # 텍스트를 인덱스로 변환 (시작/끝 토큰 추가)
        text_encoded = [self.char_to_index['<sos>']]
        text_encoded.extend(self.char_to_index[c] for c in text)
        text_encoded.append(self.char_to_index['<eos>'])
        text_encoded = torch.LongTensor(text_encoded)

        return waveform, text_encoded, len(waveform), len(text_encoded)


def collate_fn(batch):
    """배치 내 시퀀스들을 패딩"""
    waveforms, text_encoded, wav_lengths, txt_lengths = zip(*batch)

    # 오디오 패딩
    waveforms_padded = pad_sequence(waveforms, batch_first=True).unsqueeze(1)

    # 텍스트 패딩
    text_padded = pad_sequence(text_encoded, batch_first=True,
                               padding_value=0)  # <pad> 토큰은 0

    return {
        'waveforms': waveforms_padded,
        'text_encoded': text_padded,
        'wav_lengths': torch.tensor(wav_lengths),
        'txt_lengths': torch.tensor(txt_lengths)
    }


def test_pipeline():
    # 설정
    data_dir = 'D:/korean/KsponSpeech_01'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 8
    subset_size = 100  # 작은 크기로 테스트

    # 데이터셋 생성
    dataset = SubsetKsponDataset(data_dir, subset_size)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=2
    )

    # 데이터 로딩 테스트
    print("\n데이터 로딩 테스트:")
    for batch_idx, batch in enumerate(dataloader):
        if batch_idx == 0:
            print(f"배치 크기: {batch['waveforms'].size()}")
            print(f"텍스트 배치 크기: {batch['text_encoded'].size()}")
            print(f"오디오 길이: {batch['wav_lengths']}")
            print(f"텍스트 길이: {batch['txt_lengths']}")

            # 샘플 텍스트 디코딩
            idx_to_char = {v: k for k, v in dataset.char_to_index.items()}
            sample_text = batch['text_encoded'][0]
            decoded_text = ''.join(idx_to_char[idx.item()] for idx in sample_text
                                   if idx.item() != 0)  # pad 토큰 제외
            print(f"\n샘플 텍스트 디코딩: {decoded_text}")
            break

    print("\n기본 파이프라인 테스트 완료!")


if __name__ == "__main__":
    test_pipeline()