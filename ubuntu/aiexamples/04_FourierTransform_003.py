import numpy as np
import matplotlib.pyplot as plt

# 비주기적 신호 생성 (예: 가우시안 펄스)
t = np.linspace(-1, 1, 400, endpoint=False)
signal = np.exp(-t**2 * 25)

# 신호의 FFT 계산
fft_result = np.fft.fft(signal)

# 주파수 도메인의 결과 계산
freqs = np.fft.fftfreq(len(fft_result), d=t[1] - t[0])

# 시각화
plt.figure(figsize=(12, 6))

# 시간 도메인 신호
plt.subplot(2, 1, 1)
plt.plot(t, signal)
plt.title('Time Domain Signal')
plt.xlabel('Time')
plt.ylabel('Amplitude')

# 주파수 도메인 신호
plt.subplot(2, 1, 2)
plt.stem(freqs, np.abs(fft_result), 'b', markerfmt=" ", basefmt="-b")
plt.title('Frequency Domain Signal')
plt.xlabel('Frequency')
plt.ylabel('Amplitude')
plt.xlim(-30, 30) # 주파수 범위 제한

plt.tight_layout()
plt.show()
