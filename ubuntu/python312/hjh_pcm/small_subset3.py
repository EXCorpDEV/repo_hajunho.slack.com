import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import List, Tuple, Dict
from tqdm import tqdm
import random
from torch.nn.utils.rnn import pad_sequence
import matplotlib.pyplot as plt
import time  # 시간 측정을 위해 추가

class SubsetKsponDataset(Dataset):
    def __init__(self, data_dir: str, subset_size: int = 5000):
        self.data_dir = data_dir
        self.subset_size = subset_size
        # (1) 절대 경로를 사용하도록 수정
        self.file_pairs = self._get_subset_pairs()
        self.char_to_index = self._create_vocab()
        print(f"총 클래스 수 (문자 종류): {len(self.char_to_index)}")

    def _get_subset_pairs(self) -> List[Tuple[str, str]]:
        """
        여러 하위 폴더가 있어도 문제 없도록,
        각 (pcm, txt)에 대해 절대 경로를 바로 저장.
        """
        all_pairs = []
        for root, _, files in os.walk(self.data_dir):
            pcm_files = [f for f in files if f.endswith('.pcm')]
            for pcm in pcm_files:
                txt = pcm.replace('.pcm', '.txt')
                if txt in files:
                    # 절대 경로로 만들기
                    pcm_full = os.path.join(root, pcm)
                    txt_full = os.path.join(root, txt)
                    all_pairs.append((pcm_full, txt_full))

        subset_pairs = random.sample(all_pairs, min(self.subset_size, len(all_pairs)))
        print(f"선택된 데이터 수: {len(subset_pairs)}")
        return sorted(subset_pairs)

    def _create_vocab(self) -> Dict[str, int]:
        chars = {'<pad>', '<sos>', '<eos>', '<blank>'}
        for _, txt_file in self.file_pairs:
            try:
                # (2) txt_file는 절대 경로이므로 그대로 open
                with open(txt_file, 'r', encoding='cp949') as f:
                    text = f.read().strip()
                    chars.update(list(text))
            except UnicodeDecodeError:
                print(f"Warning: {txt_file} 파일을 읽는데 실패했습니다.")

        return {char: idx for idx, char in enumerate(sorted(chars))}

    def __len__(self) -> int:
        return len(self.file_pairs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int, int]:
        # (3) 이미 절대 경로이므로 그대로 사용
        pcm_path, txt_path = self.file_pairs[idx]

        # PCM 파일 로드
        with open(pcm_path, 'rb') as f:
            audio_data = np.frombuffer(f.read(), dtype=np.int16).copy()
        waveform = torch.FloatTensor(audio_data) / 32768.0

        # 텍스트 파일 로드
        with open(txt_path, 'r', encoding='cp949') as f:
            text = f.read().strip()

        # 텍스트 인코딩
        text_encoded = [self.char_to_index['<sos>']]
        text_encoded.extend(self.char_to_index[c] for c in text)
        text_encoded.append(self.char_to_index['<eos>'])
        text_encoded = torch.LongTensor(text_encoded)

        return waveform, text_encoded, len(waveform), len(text_encoded)


def collate_fn(batch):
    waveforms, text_encoded, wav_lengths, txt_lengths = zip(*batch)
    # 패딩
    waveforms_padded = pad_sequence(waveforms, batch_first=True).unsqueeze(1)  # [B, 1, T]
    text_padded = pad_sequence(text_encoded, batch_first=True, padding_value=0)  # [B, max_len]

    return {
        'waveforms': waveforms_padded,
        'text_encoded': text_padded,
        'wav_lengths': torch.tensor(wav_lengths),
        'txt_lengths': torch.tensor(txt_lengths)
    }


class SmallConformer(nn.Module):
    def __init__(self, num_classes: int, d_model: int = 144, num_layers: int = 4):
        super().__init__()

        # 프론트엔드: 간단한 CNN으로 특징 추출
        self.frontend = nn.Sequential(
            nn.Conv1d(1, d_model // 2, 10, stride=5, padding=5),
            nn.ReLU(),
            nn.Conv1d(d_model // 2, d_model, 8, stride=4, padding=4),
            nn.ReLU()
        )

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=4,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 출력 프로젝션
        self.projection = nn.Linear(d_model, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [batch, 1, time]
        x = self.frontend(x)           # [batch, d_model, time']
        x = x.transpose(1, 2)          # [batch, time', d_model]
        x = self.transformer(x)        # [batch, time', d_model]
        x = self.projection(x)         # [batch, time', num_classes]
        return x


def train_epoch(model, dataloader, optimizer, criterion, device, epoch):
    model.train()
    total_loss = 0
    epoch_start_time = time.time()
    progress_bar = tqdm(dataloader, desc=f'Epoch {epoch+1}')

    for batch_idx, batch in enumerate(progress_bar):
        optimizer.zero_grad()

        waveforms = batch['waveforms'].to(device)      # [B, 1, T_waveform]
        text_encoded = batch['text_encoded'].to(device)

        # Forward pass
        output = model(waveforms)  # [B, T_out, num_classes]

        # CTC Loss 계산을 위한 길이 정보
        # 프론트엔드가 (stride=5) → (stride=4) 총 20배 downsample, 대략: input_lengths/20
        input_lengths = batch['wav_lengths'].div(20).int()
        target_lengths = batch['txt_lengths']

        # CTC Loss
        # CTC: (T, N, C)가 필요하므로 (B, T, C) -> (T, B, C)로 transpose
        loss = criterion(
            output.transpose(0, 1).log_softmax(-1),
            text_encoded,
            input_lengths,
            target_lengths
        )

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)
        optimizer.step()

        total_loss += loss.item()
        avg_loss = total_loss / (batch_idx + 1)
        progress_bar.set_postfix({
            'loss': f'{avg_loss:.4f}',
            'time': f'{(time.time() - epoch_start_time):.1f}s'
        })

    epoch_time = time.time() - epoch_start_time
    print(f"Epoch {epoch+1} 소요 시간: {epoch_time:.2f}초")
    return avg_loss


def main():
    data_dir = 'D:/korean'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 8
    subset_size = 15000
    num_epochs = 10

    total_start_time = time.time()

    # 데이터셋
    dataset = SubsetKsponDataset(data_dir, subset_size)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=2
    )

    # 모델
    model = SmallConformer(
        num_classes=len(dataset.char_to_index),
        d_model=144,
        num_layers=4
    ).to(device)

    # 옵티마이저, 손실함수
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CTCLoss()

    print(f"\n학습 시작 - 디바이스: {device}")
    losses = []

    try:
        for epoch in range(num_epochs):
            loss_val = train_epoch(model, dataloader, optimizer, criterion, device, epoch)
            losses.append(loss_val)
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss_val:.4f}")

        total_time = time.time() - total_start_time
        print(f"\n전체 학습 소요 시간: {total_time:.2f}초 ({total_time/3600:.2f}시간)")

        # 손실 그래프
        plt.figure(figsize=(10, 5))
        plt.plot(losses)
        plt.title('Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.savefig('training_loss.png')
        plt.close()

    except KeyboardInterrupt:
        print("\n학습 중단됨")
        print(f"중단 시점까지 소요 시간: {(time.time() - total_start_time):.2f}초")

    # 모델 저장
    torch.save({
        'epoch': num_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'vocab': dataset.char_to_index,
    }, 'data/small_conformer_checkpoint.pth')

    print("학습 완료!")


if __name__ == "__main__":
    main()
