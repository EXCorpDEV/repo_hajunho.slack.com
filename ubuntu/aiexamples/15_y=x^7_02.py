import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import numpy as np

# 데이터 생성
x = np.linspace(-10, 10, 100).reshape(-1, 1) # x 값

# y = x^7 관계를 가지는 데이터 생성
y_septic = x**7

# 다항 특성 변환 (7차)
poly_features_septic = PolynomialFeatures(degree=7, include_bias=False)
x_poly_septic = poly_features_septic.fit_transform(x)

# 노이즈를 추가한 y = x^7 관계의 데이터 생성
np.random.seed(42) # 결과 일관성을 위한 시드 설정
noise = np.random.normal(0, 1e6, size=y_septic.shape) # 노이즈 생성
y_septic_noisy = y_septic + noise

# 다항 회귀 모델 학습 (노이즈 추가된 7차)
poly_reg_model_septic_noisy = LinearRegression()
poly_reg_model_septic_noisy.fit(x_poly_septic, y_septic_noisy)

# 예측값 계산 (노이즈 추가된 7차)
y_pred_septic_noisy = poly_reg_model_septic_noisy.predict(x_poly_septic)

# MSE 계산 (노이즈 추가된 7차)
mse_septic_noisy = np.mean((y_septic_noisy - y_pred_septic_noisy)**2)

# 그래프로 나타내기 (노이즈 추가된 7차)
plt.figure(figsize=(10, 6))
plt.scatter(x, y_septic_noisy, color='blue', label='Actual Points with Noise', alpha=0.6)
plt.plot(x, y_pred_septic_noisy, color='red', label='Polynomial Regression Fit')
plt.title('Polynomial Regression Fit to y = x^7 with Noise')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()

# 표로 나타내기 (노이즈 추가된 7차)
# 실제 값과 예측 값을 비교하는 작은 데이터 프레임 생성
df_comparison_septic_noisy = pd.DataFrame({'x': x.flatten(), 'Actual y': y_septic_noisy.flatten(), 'Predicted y': y_pred_septic_noisy.flatten()})
df_comparison_septic_noisy = df_comparison_septic_noisy.head(50) # 처음 50개의 값만 표시

print(mse_septic_noisy, '\n', df_comparison_septic_noisy)
