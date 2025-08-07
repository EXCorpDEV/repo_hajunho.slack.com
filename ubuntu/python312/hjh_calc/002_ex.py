import numpy as np
import matplotlib.pyplot as plt

# x 범위
x = np.linspace(-2, 2, 400)

# 함수 정의
y_2 = 2 ** x
y_3 = 3 ** x
y_e = np.exp(x)

# 미분 결과
dy_2 = y_2 * np.log(2)
dy_3 = y_3 * np.log(3)
dy_e = y_e  # 그대로!

# 그래프 스타일
plt.figure(figsize=(100, 60))

# 원 함수
plt.plot(x, y_2, label='2^x', linestyle='--', color='blue')
plt.plot(x, y_3, label='3^x', linestyle='--', color='green')
plt.plot(x, y_e, label='e^x', linestyle='--', color='red')

# 미분 함수
plt.plot(x, dy_2, label="d/dx 2^x", color='blue')
plt.plot(x, dy_3, label="d/dx 3^x", color='green')
plt.plot(x, dy_e, label="d/dx e^x", color='red')

# 스타일 설정
plt.axhline(0, color='black', linewidth=0.5)
plt.title("지수함수와 그 미분 결과 비교")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
