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

# 다항 회귀 모델 학습 (7차)
poly_reg_model_septic = LinearRegression()
poly_reg_model_septic.fit(x_poly_septic, y_septic)

# 예측값 계산 (7차)
y_pred_septic = poly_reg_model_septic.predict(x_poly_septic)

# MSE 계산 (7차)
mse_septic = np.mean((y_septic - y_pred_septic)**2)

# 그래프로 나타내기 (7차)
plt.figure(figsize=(10, 6))
plt.scatter(x, y_septic, color='blue', label='Actual Points', alpha=0.6)
plt.plot(x, y_pred_septic, color='red', label='Polynomial Regression Fit')
plt.title('Polynomial Regression Fit to y = x^7')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()

# 표로 나타내기 (7차)
# 실제 값과 예측 값을 비교하는 작은 데이터 프레임 생성
df_comparison_septic = pd.DataFrame({'x': x.flatten(), 'Actual y': y_septic.flatten(), 'Predicted y': y_pred_septic.flatten()})
df_comparison_septic = df_comparison_septic.head(10) # 처음 10개의 값만 표시

print(mse_septic,'\n', df_comparison_septic)
