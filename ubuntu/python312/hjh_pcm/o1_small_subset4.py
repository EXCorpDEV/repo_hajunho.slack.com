import os
import random
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
import matplotlib.pyplot as plt
import torchaudio

# -------------------
# 1) Conformer 블록 예시
# -------------------
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConformerBlock(nn.Module):
    """
    간략히 Conformer 블록 구조를 예시로 구현:
    - Macaron FeedForward (전단)
    - Multi-Head Self-Attention
    - Convolution Module (GLU 사용)
    - Macaron FeedForward (후단)
    - LayerNorm 등
    """
    def __init__(self, d_model, n_heads=4, ff_multiplier=4, conv_kernel=7, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        # --- 1) Macaron FeedForward #1 ---
        self.ff1 = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * ff_multiplier),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * ff_multiplier, d_model),
            nn.Dropout(dropout),
        )

        # --- 2) Self-Attention ---
        self.ln_attn = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, dropout=dropout, batch_first=True)

        # --- 3) Convolution Module ---
        self.ln_conv = nn.LayerNorm(d_model)
        # pointwise_conv1: (d_model -> 2*d_model)
        self.pointwise_conv1 = nn.Conv1d(d_model, 2*d_model, kernel_size=1)
        # depthwise_conv: (in=d_model -> out=d_model)를 위해, GLU 후 채널이 d_model이 되도록 설계
        #   하지만 GLU를 pointwise_conv1 -> depthwise_conv 사이에 넣기 위해서는
        #   depthwise_conv in/out도 d_model이 되어야 합니다.
        #   아래에서는 GLU를 pointwise_conv1 직후에 적용.
        self.depthwise_conv = nn.Conv1d(d_model, d_model, kernel_size=conv_kernel,
                                        padding=conv_kernel//2, groups=d_model)
        self.batchnorm = nn.BatchNorm1d(d_model)
        self.act = nn.ReLU()  # 예: ReLU, SiLU, Swish 등
        # pointwise_conv2: (d_model -> d_model)
        self.pointwise_conv2 = nn.Conv1d(d_model, d_model, kernel_size=1)

        # --- 4) Macaron FeedForward #2 ---
        self.ff2 = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * ff_multiplier),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * ff_multiplier, d_model),
            nn.Dropout(dropout),
        )

        self.ln_out = nn.LayerNorm(d_model)

    def forward(self, x):
        """
        x: [B, T, d_model]
        반환: [B, T, d_model]
        """
        # --- FeedForward #1 (Macaron) ---
        x = x + 0.5 * self.ff1(x)

        # --- Multi-Head Self-Attention ---
        x_norm = self.ln_attn(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)  # batch_first=True
        x = x + attn_out

        # --- Convolution Module ---
        residual = x
        x_conv = self.ln_conv(x)
        # (B, T, d_model) -> (B, d_model, T)
        x_conv = x_conv.transpose(1, 2)

        # 1) pointwise_conv1: d_model -> 2*d_model
        x_conv = self.pointwise_conv1(x_conv)  # [B, 2*d_model, T]

        # 2) GLU: 2*d_model => split => d_model
        xA, xB = x_conv.chunk(2, dim=1)  # (2*d_model) -> (d_model, d_model)
        x_conv = xA * torch.sigmoid(xB)  # => [B, d_model, T]

        # 3) depthwise_conv: d_model -> d_model
        x_conv = self.depthwise_conv(x_conv)   # [B, d_model, T]
        x_conv = self.batchnorm(x_conv)
        x_conv = self.act(x_conv)

        # 4) pointwise_conv2: d_model -> d_model
        x_conv = self.pointwise_conv2(x_conv)  # [B, d_model, T]

        # (B, d_model, T) -> (B, T, d_model)
        x_conv = x_conv.transpose(1, 2)

        # residual
        x = residual + x_conv

        # --- FeedForward #2 (Macaron) ---
        x = x + 0.5 * self.ff2(x)

        # --- 최종 LayerNorm ---
        x = self.ln_out(x)
        return x

# -------------------
# 2) 실제 Conformer Encoder
# -------------------
class SmallConformer(nn.Module):
    def __init__(self, num_classes: int, d_model: int = 144, num_layers: int = 4, n_heads: int = 4):
        super().__init__()

        # MelSpectrogram으로 바꿀 경우, 채널=1이 아닌 [batch, freq, time] 형태가 일반적이므로
        # 여기서는 conv로 downsampling 하기보다는, 그냥 2D -> 1D처럼 flatten하거나
        # 또는 Conformer가 2D 입력을 그대로 처리하도록 만들 수도 있음.
        # 예시는 "time축" 기준으로만 self-attention을 적용한다고 가정해서,
        # (B, freq, time) -> (B, time, freq)로 변환 후 d_model로 projection

        self.d_model = d_model

        # 간단히, MelSpectrogram 계층(실시간 변환) 추가
        # sample_rate, n_mels, n_fft 등은 실제 데이터셋에 맞게 조정
        self.mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=16000,
            n_mels=80,
            n_fft=400,
            hop_length=160
        )
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB()

        # 입력 차원 = n_mels(80) -> d_model
        self.input_linear = nn.Linear(80, d_model)

        # Conformer Blocks
        self.conformer_blocks = nn.ModuleList([
            ConformerBlock(d_model, n_heads=n_heads, conv_kernel=7, ff_multiplier=4, dropout=0.1)
            for _ in range(num_layers)
        ])

        # 최종 투사
        self.fc_out = nn.Linear(d_model, num_classes)

    def forward(self, x, lengths):
        """
        x: [B, 1, T_waveform]  (raw audio)
        lengths: 실제 오디오 샘플 길이
        """
        # 1) MelSpectrogram 변환
        #  - batch 별로 loop 돌려야 할 수도 있음 (torchaudio는 배치 연산 지원)
        #  - 간단히 for문 사용(배치가 크면 성능 저하 가능)
        #  - 아래는 데모를 위해 간략히 작성
        batch_mel = []
        mel_lengths = []
        for i in range(x.size(0)):
            wav_i = x[i, 0, :lengths[i]]  # 실제 길이만큼 slice
            mel_i = self.mel_spectrogram(wav_i)
            mel_i = self.amplitude_to_db(mel_i)
            # mel_i shape: [n_mels, T_spec]
            mel_len = mel_i.size(1)
            batch_mel.append(mel_i.transpose(0,1))  # -> [T_spec, n_mels]
            mel_lengths.append(mel_len)

        # pad_sequence로 [B, T_spec, n_mels] 맞춤
        mel_padded = pad_sequence(batch_mel, batch_first=True, padding_value=-80.0)
        # 여기서 mel_padded.shape = [B, T_spec_max, n_mels]

        # 2) Linear로 d_model 투영
        out = self.input_linear(mel_padded)  # [B, T_spec, d_model]

        # 3) Conformer blocks
        for block in self.conformer_blocks:
            out = block(out)  # [B, T_spec, d_model]

        # 4) Output projection
        logits = self.fc_out(out)  # [B, T_spec, num_classes]

        return logits, mel_lengths  # CTC에서 입력 길이는 mel_lengths를 사용

# -------------------
# 3) Dataset & Collate
# -------------------
class SubsetKsponDataset(Dataset):
    def __init__(self, data_dir: str, subset_size: int = 5000):
        self.data_dir = data_dir
        self.subset_size = subset_size
        self.file_pairs = self._get_subset_pairs()

        # 여기서 'vocab'을 구성할 때, CTC blank 토큰과 pad 토큰을 잘 구분해줍니다.
        self.char_to_index = self._create_vocab()

        print(f"총 클래스 수 (문자 종류): {len(self.char_to_index)}")

    def _get_subset_pairs(self):
        all_pairs = []
        for root, _, files in os.walk(self.data_dir):
            pcm_files = [f for f in files if f.endswith('.pcm')]
            for pcm in pcm_files:
                txt = pcm.replace('.pcm', '.txt')
                if txt in files:
                    rel_path = os.path.relpath(root, self.data_dir)
                    all_pairs.append((os.path.join(rel_path, pcm),
                                      os.path.join(rel_path, txt)))

        subset_pairs = random.sample(all_pairs, min(self.subset_size, len(all_pairs)))
        print(f"선택된 데이터 수: {len(subset_pairs)}")
        return sorted(subset_pairs)

    def _create_vocab(self):
        """
        예시:
        0 -> <blank>  (CTC blank)
        그 외 실제 문자들 (한글/영문/기타) 1 ~ ...
        """
        # 우선 blank 토큰만 미리 추가
        chars = ['<blank>']

        # 실제 텍스트에서 문자 추출
        unique_chars = set()
        for _, txt_file in self.file_pairs:
            try:
                with open(os.path.join(self.data_dir, txt_file), 'r', encoding='cp949') as f:
                    text = f.read().strip()
                    # 여기서 <sos>, <eos> 따로 안 쓰고, 그냥 순수 문자만 수집
                    unique_chars.update(list(text))
            except UnicodeDecodeError:
                print(f"Warning: {txt_file} 파일을 읽는데 실패했습니다.")

        chars.extend(sorted(unique_chars))

        # {문자: 인덱스}
        return {ch: i for i, ch in enumerate(chars)}

    def __len__(self):
        return len(self.file_pairs)

    def __getitem__(self, idx: int):
        pcm_path = os.path.join(self.data_dir, self.file_pairs[idx][0])
        txt_path = os.path.join(self.data_dir, self.file_pairs[idx][1])

        # PCM load
        with open(pcm_path, 'rb') as f:
            audio_data = np.frombuffer(f.read(), dtype=np.int16).copy()
        waveform = torch.FloatTensor(audio_data) / 32768.0

        # text load
        with open(txt_path, 'r', encoding='cp949') as f:
            text = f.read().strip()

        # text -> index
        text_encoded = [self.char_to_index[ch] for ch in text if ch in self.char_to_index]
        # torch로 변환
        text_encoded = torch.LongTensor(text_encoded)

        return waveform, text_encoded, len(waveform), len(text_encoded)

def collate_fn(batch):
    """CTC를 위한 collate"""
    waveforms, text_encoded, wav_lengths, txt_lengths = zip(*batch)

    # waveforms를 pad: [B, 1, T_waveform]
    waveforms_padded = pad_sequence(waveforms, batch_first=True).unsqueeze(1)
    wav_lengths = torch.tensor(wav_lengths, dtype=torch.long)

    # 텍스트도 pad: 단, 여기서는 CTC Loss에 넣을 때 pad는 사실 큰 의미가 없으니
    # 그냥 -1 같은 걸로 패딩하고, CTC 계산할 때 target_lengths로 구분.
    # 또는 padding_value=0(=<blank>)로 할 수도 있는데, 혼동이 없도록 주의.
    # 여기서는 일단 -1로 패딩
    text_padded = pad_sequence(text_encoded, batch_first=True, padding_value=-1)
    txt_lengths = torch.tensor(txt_lengths, dtype=torch.long)

    return {
        'waveforms': waveforms_padded,
        'text_encoded': text_padded,
        'wav_lengths': wav_lengths,
        'txt_lengths': txt_lengths
    }

# -------------------
# 4) Train 함수
# -------------------
def train_epoch(model, dataloader, optimizer, criterion, device, epoch, blank_idx=0):
    model.train()
    total_loss = 0
    epoch_start_time = time.time()
    progress_bar = tqdm(dataloader, desc=f'Epoch {epoch+1}')

    for batch_idx, batch in enumerate(progress_bar):
        optimizer.zero_grad()

        waveforms = batch['waveforms'].to(device)
        text_encoded = batch['text_encoded'].to(device)
        wav_lengths = batch['wav_lengths'].to(device)
        txt_lengths = batch['txt_lengths'].to(device)

        # forward
        logits, feat_lengths = model(waveforms, wav_lengths)  # [B, T_spec, num_classes], feat_lengths = list of int

        # CTC Loss는 (T, N, C) shape을 기대하므로 transpose 필요
        # 또한, feat_lengths는 파이썬 list이므로 tensor로 변환 필요
        # (단, pad_sequence 후 길이는 maxLen이 될 것이고, 각 샘플 별로 실제 길이는 feat_lengths[i])
        # -> 모듈화된 front-end에서 정확한 타임스텝 수를 계산해야 함
        # 여기서는 feat_lengths가 모델 forward에서 추출된 mel_lengths를 리스트로 반환하므로
        # 아래처럼 해주면 됨
        max_len = logits.size(1)
        # 만들어진 feat_length 텐서
        input_lengths = torch.tensor(feat_lengths, dtype=torch.long).to(device)

        # text에 들어있는 -1 padding은 CTC에서 처리 불가능하므로
        # 실제 target_lengths만큼 slicing하거나,
        # 혹은 (target == -1)을 마스킹해서 빼거나 해야 함
        # PyTorch CTC Loss는 그런 마스킹 기능이 없으므로
        # 보통 padding_value를 blank로 두고, target_lengths를 정확히 설정.
        # 여기서는 단순화 위해 padding_value = blank_idx = 0으로 하는 편이 나음

        # 그래도 -1 패딩을 쓴 경우, 아래처럼 masking해서 flatten 후 CTC에 넣는 방법도 있지만 복잡.
        # 여기서는 그냥 "padding_value=blank_idx"로 가정하겠습니다...
        # 즉, 만약 collate_fn에서 padding_value=0(=<blank>)으로 했다면 이런 처리가 필요 없음.

        # (데모 용으로, -1을 blank_idx로 교체)
        text_encoded = torch.where(text_encoded == -1,
                                   torch.tensor(blank_idx, device=device),
                                   text_encoded)

        # CTC Loss
        # logits: [B, T, C] -> [T, B, C]
        log_probs = logits.log_softmax(dim=-1).transpose(0, 1)
        loss = criterion(log_probs, text_encoded, input_lengths, txt_lengths)

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        total_loss += loss.item()
        avg_loss = total_loss / (batch_idx + 1)
        progress_bar.set_postfix({
            'loss': f'{avg_loss:.4f}',
            'time': f'{(time.time() - epoch_start_time):.1f}s'
        })

    epoch_time = time.time() - epoch_start_time
    print(f"Epoch {epoch + 1} 소요 시간: {epoch_time:.2f}초")
    return total_loss / len(dataloader)

# -------------------
# 5) main
# -------------------
def main():
    data_dir = 'D:/korean/KsponSpeech_01'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 4
    subset_size = 5000
    num_epochs = 10

    total_start_time = time.time()

    # Dataset & DataLoader
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
        num_classes=len(dataset.char_to_index),  # 첫 index=0이 <blank>
        d_model=144,
        num_layers=4,
        n_heads=4
    ).to(device)

    # CTC Loss(blank=0) - 위에서 <blank>를 0번 인덱스로 했으므로
    criterion = nn.CTCLoss(blank=0, reduction='mean', zero_infinity=True)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # 학습 루프
    losses = []
    print(f"\n학습 시작 - 디바이스: {device}")
    try:
        for epoch in range(num_epochs):
            loss_val = train_epoch(model, dataloader, optimizer, criterion, device, epoch, blank_idx=0)
            losses.append(loss_val)
            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss_val:.4f}")

        total_time = time.time() - total_start_time
        print(f"\n전체 학습 소요 시간: {total_time:.2f}초 ({total_time / 3600:.2f}시간)")

        # 손실 그래프
        plt.figure(figsize=(10, 5))
        plt.plot(losses, marker='o')
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
    }, 'small_conformer_checkpoint.pth')

    print("학습 완료!")

if __name__ == "__main__":
    main()

