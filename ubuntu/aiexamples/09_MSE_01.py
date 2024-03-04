import numpy as np

# 예측값과 실제값
y_true = np.array([3, -0.5, 2, 7])
y_pred = np.array([2.5, 0.0, 2, 8])

# 평균 제곱 오차 (MSE) 계산
mse = np.mean((y_true - y_pred)**2)

print(mse)
