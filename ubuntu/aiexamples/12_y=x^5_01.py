import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import numpy as np

# 데이터 생성
x = np.linspace(-10, 10, 100).reshape(-1, 1) # x 값

# y = x^5 관계를 가지는 데이터 생성
y_quintic = x**5

# 다항 특성 변환 (5차)
poly_features_quintic = PolynomialFeatures(degree=5, include_bias=False)
x_poly_quintic = poly_features_quintic.fit_transform(x)

# 다항 회귀 모델 학습 (5차)
poly_reg_model_quintic = LinearRegression()
poly_reg_model_quintic.fit(x_poly_quintic, y_quintic)

# 예측값 계산 (5차)
y_pred_quintic = poly_reg_model_quintic.predict(x_poly_quintic)

# MSE 계산 (5차)
mse_quintic = np.mean((y_quintic - y_pred_quintic)**2)

# 그래프로 나타내기 (5차)
plt.figure(figsize=(10, 6))
plt.scatter(x, y_quintic, color='blue', label='Actual Points', alpha=0.6)
plt.plot(x, y_pred_quintic, color='red', label='Polynomial Regression Fit')
plt.title('Polynomial Regression Fit to y = x^5')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()

# 표로 나타내기 (5차)
# 실제 값과 예측 값을 비교하는 작은 데이터 프레임 생성
df_comparison_quintic = pd.DataFrame({'x': x.flatten(), 'Actual y': y_quintic.flatten(), 'Predicted y': y_pred_quintic.flatten()})
df_comparison_quintic = df_comparison_quintic.head(10) # 처음 10개의 값만 표시

print(df_comparison_quintic)
print(mse_quintic)
