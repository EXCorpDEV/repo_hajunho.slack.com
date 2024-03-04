import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import numpy as np

# 데이터 생성
x = np.linspace(-10, 10, 100).reshape(-1, 1) # x 값
y = x**2 # y = x^2

# 다항 특성 변환
poly_features = PolynomialFeatures(degree=2, include_bias=False)
x_poly = poly_features.fit_transform(x)

# 다항 회귀 모델 학습
poly_reg_model = LinearRegression()
poly_reg_model.fit(x_poly, y)

# 예측값 계산
y_pred = poly_reg_model.predict(x_poly)

# 그래프로 나타내기
plt.figure(figsize=(10, 6))
plt.scatter(x, y, color='blue', label='Actual Points', alpha=0.6)
plt.plot(x, y_pred, color='red', label='Polynomial Regression Fit')
plt.title('Polynomial Regression Fit to y = x^2')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()

# 표로 나타내기
# 실제 값과 예측 값을 비교하는 작은 데이터 프레임 생성
df_comparison = pd.DataFrame({'x': x.flatten(), 'Actual y': y.flatten(), 'Predicted y': y_pred.flatten()})
df_comparison = df_comparison.head(10) # 처음 10개의 값만 표시
print(df_comparison)
