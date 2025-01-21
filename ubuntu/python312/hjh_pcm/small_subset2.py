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


class SubsetKsponDataset(Dataset):
    def __init__(self, data_dir: str, subset_size: int = 1000):
        self.data_dir = data_dir
        self.subset_size = subset_size
        self.file_pairs = self._get_subset_pairs()
        self.char_to_index = self._create_vocab()
        print(f"총 클래스 수 (문자 종류): {len(self.char_to_index)}")

    def _get_subset_pairs(self) -> List[Tuple[str, str]]:
        all_pairs = []
        for root, _, files in os.walk(self.data_dir):
            pcm_files = {f for f in files if f.endswith('.pcm')}
            for pcm in pcm_files:
                txt = pcm.replace('.pcm', '.txt')
                if txt in files:
                    rel_path = os.path.relpath(root, self.data_dir)
                    all_pairs.append((os.path.join(rel_path, pcm),
                                      os.path.join(rel_path, txt)))

        subset_pairs = random.sample(all_pairs, min(self.subset_size, len(all_pairs)))
        print(f"선택된 데이터 수: {len(subset_pairs)}")
        return sorted(subset_pairs)

    def _create_vocab(self) -> Dict[str, int]:
        chars = {'<pad>', '<sos>', '<eos>', '<blank>'}
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

        # PCM 파일 로드 (numpy array copy로 경고 해결)
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
    waveforms_padded = pad_sequence(waveforms, batch_first=True).unsqueeze(1)
    text_padded = pad_sequence(text_encoded, batch_first=True, padding_value=0)

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
        x = self.frontend(x)
        x = x.transpose(1, 2)  # [batch, time, channels]
        x = self.transformer(x)
        x = self.projection(x)
        return x


def train_epoch(model, dataloader, optimizer, criterion, device, epoch):
    model.train()
    total_loss = 0
    progress_bar = tqdm(dataloader, desc=f'Epoch {epoch}')

    for batch_idx, batch in enumerate(progress_bar):
        optimizer.zero_grad()

        # 데이터를 GPU로
        waveforms = batch['waveforms'].to(device)
        text_encoded = batch['text_encoded'].to(device)

        # Forward pass
        output = model(waveforms)

        # CTC Loss 계산을 위한 길이 정보
        input_lengths = batch['wav_lengths'].div(20).int()  # 프론트엔드의 총 stride
        target_lengths = batch['txt_lengths']

        # Loss 계산
        loss = criterion(
            output.transpose(0, 1).log_softmax(-1),
            text_encoded,
            input_lengths,
            target_lengths
        )

        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)
        optimizer.step()

        total_loss += loss.item()
        avg_loss = total_loss / (batch_idx + 1)
        progress_bar.set_postfix({'loss': f'{avg_loss:.4f}'})

    return total_loss / len(dataloader)


def main():
    # 설정
    data_dir = 'D:/korean/KsponSpeech_01'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 8
    subset_size = 1000  # 작은 크기로 시작
    num_epochs = 10

    # 데이터셋 준비
    dataset = SubsetKsponDataset(data_dir, subset_size)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=2
    )

    # 모델 설정
    model = SmallConformer(
        num_classes=len(dataset.char_to_index),
        d_model=144,  # 작은 모델 사이즈
        num_layers=4
    ).to(device)

    # 옵티마이저와 손실 함수
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CTCLoss()

    # 학습 루프
    losses = []
    print(f"\n학습 시작 - 디바이스: {device}")

    try:
        for epoch in range(num_epochs):
            loss = train_epoch(model, dataloader, optimizer, criterion, device, epoch)
            losses.append(loss)
            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss:.4f}")

        # 손실 그래프 그리기
        plt.figure(figsize=(10, 5))
        plt.plot(losses)
        plt.title('Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.savefig('training_loss.png')
        plt.close()

    except KeyboardInterrupt:
        print("\n학습 중단됨")

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