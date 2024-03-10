import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import numpy as np

# 데이터 생성
x = np.linspace(-10, 10, 100).reshape(-1, 1) # x 값

# y = x^6 관계를 가지는 데이터 생성
y_sextic = x**6

# 다항 특성 변환 (6차)
poly_features_sextic = PolynomialFeatures(degree=6, include_bias=False)
x_poly_sextic = poly_features_sextic.fit_transform(x)

# 다항 회귀 모델 학습 (6차)
poly_reg_model_sextic = LinearRegression()
poly_reg_model_sextic.fit(x_poly_sextic, y_sextic)

# 예측값 계산 (6차)
y_pred_sextic = poly_reg_model_sextic.predict(x_poly_sextic)

# MSE 계산 (6차)
mse_sextic = np.mean((y_sextic - y_pred_sextic)**2)

# 그래프로 나타내기 (6차)
plt.figure(figsize=(10, 6))
plt.scatter(x, y_sextic, color='blue', label='Actual Points', alpha=0.6)
plt.plot(x, y_pred_sextic, color='red', label='Polynomial Regression Fit')
plt.title('Polynomial Regression Fit to y = x^6')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()

# 표로 나타내기 (6차)
# 실제 값과 예측 값을 비교하는 작은 데이터 프레임 생성
df_comparison_sextic = pd.DataFrame({'x': x.flatten(), 'Actual y': y_sextic.flatten(), 'Predicted y': y_pred_sextic.flatten()})
df_comparison_sextic = df_comparison_sextic.head(10) # 처음 10개의 값만 표시

print(f'{mse_sextic} \n {df_comparison_sextic}')
