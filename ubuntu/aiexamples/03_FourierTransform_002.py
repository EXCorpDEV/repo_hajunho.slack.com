import numpy as np
import matplotlib.pyplot as plt

# 시간 도메인의 신호 생성 (예: 사인파와 코사인파의 합성, 주파수를 다르게 하여 더 명확하게 구분)
t = np.linspace(0, 1, 400, endpoint=False)
# 주파수를 조정하여 각각의 성분이 더 명확하게 나타나도록 함
signal = np.sin(2 * np.pi * 5 * t) + 0.5 * np.cos(2 * np.pi * 15 * t)

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
plt.xlim(-20, 20) # 주파수 범위 제한

plt.tight_layout()
plt.show()
