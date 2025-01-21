import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchaudio
from torch.nn.utils.rnn import pad_sequence


# Conformer 블록 클래스는 모델 구조를 위해 필요
class ConformerBlock(nn.Module):
    def __init__(self, d_model, n_heads=4, ff_multiplier=4, conv_kernel=7, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        # Macaron FeedForward #1
        self.ff1 = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * ff_multiplier),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * ff_multiplier, d_model),
            nn.Dropout(dropout),
        )

        # Self-Attention
        self.ln_attn = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, dropout=dropout, batch_first=True)

        # Convolution Module
        self.ln_conv = nn.LayerNorm(d_model)
        self.pointwise_conv1 = nn.Conv1d(d_model, 2 * d_model, kernel_size=1)
        self.depthwise_conv = nn.Conv1d(d_model, d_model, kernel_size=conv_kernel,
                                        padding=conv_kernel // 2, groups=d_model)
        self.batchnorm = nn.BatchNorm1d(d_model)
        self.act = nn.ReLU()
        self.pointwise_conv2 = nn.Conv1d(d_model, d_model, kernel_size=1)

        # Macaron FeedForward #2
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
        x = x + 0.5 * self.ff1(x)
        x_norm = self.ln_attn(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_out

        residual = x
        x_conv = self.ln_conv(x)
        x_conv = x_conv.transpose(1, 2)
        x_conv = self.pointwise_conv1(x_conv)
        xA, xB = x_conv.chunk(2, dim=1)
        x_conv = xA * torch.sigmoid(xB)
        x_conv = self.depthwise_conv(x_conv)
        x_conv = self.batchnorm(x_conv)
        x_conv = self.act(x_conv)
        x_conv = self.pointwise_conv2(x_conv)
        x_conv = x_conv.transpose(1, 2)
        x = residual + x_conv
        x = x + 0.5 * self.ff2(x)
        x = self.ln_out(x)
        return x


# 모델 클래스도 필요
class SmallConformer(nn.Module):
    def __init__(self, num_classes: int, d_model: int = 144, num_layers: int = 4, n_heads: int = 4):
        super().__init__()
        self.d_model = d_model
        self.mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=16000,
            n_mels=80,
            n_fft=400,
            hop_length=160
        )
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB()
        self.input_linear = nn.Linear(80, d_model)
        self.conformer_blocks = nn.ModuleList([
            ConformerBlock(d_model, n_heads=n_heads, conv_kernel=7, ff_multiplier=4, dropout=0.1)
            for _ in range(num_layers)
        ])
        self.fc_out = nn.Linear(d_model, num_classes)

    def forward(self, x, lengths=None):
        if lengths is None:
            lengths = [x.size(-1)]
        batch_mel = []
        mel_lengths = []
        for i in range(x.size(0)):
            wav_i = x[i, 0, :lengths[i]]
            mel_i = self.mel_spectrogram(wav_i)
            mel_i = self.amplitude_to_db(mel_i)
            mel_len = mel_i.size(1)
            batch_mel.append(mel_i.transpose(0, 1))
            mel_lengths.append(mel_len)

        mel_padded = pad_sequence(batch_mel, batch_first=True, padding_value=-80.0)
        out = self.input_linear(mel_padded)

        for block in self.conformer_blocks:
            out = block(out)

        logits = self.fc_out(out)
        return logits, mel_lengths


def transcribe_audio(pcm_path, model, char_to_index, device='cuda'):
    # PCM 파일 로드
    with open(pcm_path, 'rb') as f:
        audio_data = np.frombuffer(f.read(), dtype=np.int16).copy()
    waveform = torch.FloatTensor(audio_data) / 32768.0

    # 모델 입력을 위한 차원 추가
    waveform = waveform.unsqueeze(0).unsqueeze(0).to(device)

    # 추론
    model.eval()
    with torch.no_grad():
        output, _ = model(waveform)
        output = torch.argmax(output, dim=-1)[0]  # 배치 차원 제거
        pred_sequence = output.cpu().numpy()

    # 디코딩 (CTC collapse)
    index_to_char = {idx: char for char, idx in char_to_index.items()}
    prev_char = None
    decoded_text = []

    for idx in pred_sequence:
        char = index_to_char[idx]
        if char not in ['<blank>'] and char != prev_char:
            decoded_text.append(char)
            prev_char = char

    return ''.join(decoded_text)


def main():
    # 설정
    checkpoint_path = 'small_conformer_checkpoint.pth'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    test_pcm_path = './KsponSpeech_000001.pcm'

    print(f"디바이스: {device}")

    # 모델과 vocab 로드
    print("모델 로딩 중...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = SmallConformer(
        num_classes=len(checkpoint['vocab']),
        d_model=144,
        num_layers=4,
        n_heads=4
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print("모델 로딩 완료")

    # 음성 변환
    print("\n음성 변환 중...")
    try:
        transcription = transcribe_audio(test_pcm_path, model, checkpoint['vocab'], device)
        print("\n변환 결과:")
        print(f"입력 파일: {test_pcm_path}")
        print(f"변환된 텍스트: {transcription}")
    except Exception as e:
        print(f"에러 발생: {e}")


if __name__ == "__main__":
    main()