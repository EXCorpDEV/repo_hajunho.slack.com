import matplotlib.pyplot as plt
import numpy as np

# y = 2x 방정식을 x 값의 범위에서 정의합니다.
x = np.linspace(-10, 10, 400)  # x 값의 범위를 -10에서 10까지 400개의 점으로 나눕니다.
y = 2 * x  # y 값을 2x로 계산합니다.

# 그래프를 그립니다.
plt.figure(figsize=(8, 6))  # 그래프의 크기를 설정합니다. (가로 8, 세로 6)
plt.plot(x, y, label='y = 2x')  # x와 y 값을 사용하여 그래프를 그립니다. 라벨을 'y = 2x'로 설정합니다.
plt.title('Graph of y = 2x')  # 그래프의 제목을 설정합니다.
plt.xlabel('x')  # x축의 라벨을 설정합니다.
plt.ylabel('y')  # y축의 라벨을 설정합니다.
plt.axhline(0, color='black', linewidth=0.5)  # y=0 축에 검은색 선을 그립니다. (두께 0.5)
plt.axvline(0, color='black', linewidth=0.5)  # x=0 축에 검은색 선을 그립니다. (두께 0.5)
plt.grid(color='gray', linestyle='--', linewidth=0.5)  # 회색 점선으로 그리드를 설정합니다. (두께 0.5)
plt.legend()  # 범례를 표시합니다.

# 그래프를 화면에 출력합니다.
plt.show()  # 그래프를 보여줍니다.
