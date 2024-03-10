import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import numpy as np

# 데이터 생성
x = np.linspace(-10, 10, 100).reshape(-1, 1) # x 값

# y = x^4 관계를 가지는 데이터 생성
y_quartic = x**4

# 다항 특성 변환 (4차)
poly_features_quartic = PolynomialFeatures(degree=4, include_bias=False)
x_poly_quartic = poly_features_quartic.fit_transform(x)

# 다항 회귀 모델 학습 (4차)
poly_reg_model_quartic = LinearRegression()
poly_reg_model_quartic.fit(x_poly_quartic, y_quartic)

# 예측값 계산 (4차)
y_pred_quartic = poly_reg_model_quartic.predict(x_poly_quartic)

# MSE 계산 (4차)
mse_quartic = np.mean((y_quartic - y_pred_quartic)**2)

# 그래프로 나타내기 (4차)
plt.figure(figsize=(10, 6))
plt.scatter(x, y_quartic, color='blue', label='Actual Points', alpha=0.6)
plt.plot(x, y_pred_quartic, color='red', label='Polynomial Regression Fit')
plt.title('Polynomial Regression Fit to y = x^4')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()

# 표로 나타내기 (4차)
# 실제 값과 예측 값을 비교하는 작은 데이터 프레임 생성
df_comparison_quartic = pd.DataFrame({'x': x.flatten(), 'Actual y': y_quartic.flatten(), 'Predicted y': y_pred_quartic.flatten()})
df_comparison_quartic = df_comparison_quartic.head(10) # 처음 10개의 값만 표시

print(mse_quartic, df_comparison_quartic)
