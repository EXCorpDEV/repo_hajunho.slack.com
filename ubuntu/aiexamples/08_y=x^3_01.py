import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import numpy as np

# 데이터 생성
x = np.linspace(-10, 10, 100).reshape(-1, 1) # x 값
# y = x^3 관계를 가지는 데이터 생성
y_cubic = x**3

# 다항 특성 변환 (3차)
poly_features_cubic = PolynomialFeatures(degree=3, include_bias=False)
x_poly_cubic = poly_features_cubic.fit_transform(x)

# 다항 회귀 모델 학습 (3차)
poly_reg_model_cubic = LinearRegression()
poly_reg_model_cubic.fit(x_poly_cubic, y_cubic)

# 예측값 계산 (3차)
y_pred_cubic = poly_reg_model_cubic.predict(x_poly_cubic)

# 그래프로 나타내기 (3차)
plt.figure(figsize=(10, 6))
plt.scatter(x, y_cubic, color='blue', label='Actual Points', alpha=0.6)
plt.plot(x, y_pred_cubic, color='red', label='Polynomial Regression Fit')
plt.title('Polynomial Regression Fit to y = x^3')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()

# 표로 나타내기 (3차)
# 실제 값과 예측 값을 비교하는 작은 데이터 프레임 생성
df_comparison_cubic = pd.DataFrame({'x': x.flatten(), 'Actual y': y_cubic.flatten(), 'Predicted y': y_pred_cubic.flatten()})
df_comparison_cubic = df_comparison_cubic.head(10) # 처음 10개의 값만 표시
print(df_comparison_cubic)
