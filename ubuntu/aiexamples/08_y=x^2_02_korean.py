import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import numpy as np

# 데이터 생성
x = np.linspace(-10, 10, 100).reshape(-1, 1)  # x 값을 -10에서 10까지 100개의 점으로 나누어 생성하고 열 벡터로 변환합니다.
y = x**2  # y 값을 x의 제곱으로 계산합니다.

# 다항 특성 변환
poly_features = PolynomialFeatures(degree=2, include_bias=False)  # 2차 다항식 특성을 생성하는 객체를 만듭니다. 상수항을 포함하지 않습니다.
x_poly = poly_features.fit_transform(x)  # x 데이터를 다항 특성으로 변환합니다.

# 다항 회귀 모델 학습
poly_reg_model = LinearRegression()  # 선형 회귀 모델 객체를 생성합니다.
poly_reg_model.fit(x_poly, y)  # 다항 특성으로 변환된 x와 y를 사용하여 모델을 학습시킵니다.

# 예측값 계산
y_pred = poly_reg_model.predict(x_poly)  # 학습된 모델을 사용하여 예측값을 계산합니다.

# 그래프로 나타내기
plt.figure(figsize=(10, 6))  # 그래프의 크기를 설정합니다. (가로 10, 세로 6)
plt.scatter(x, y, color='blue', label='Actual Points', alpha=0.6)  # 실제 데이터를 파란색 점으로 나타냅니다. (투명도 0.6)
plt.plot(x, y_pred, color='red', label='Polynomial Regression Fit')  # 예측된 값을 빨간색 선으로 나타냅니다.
plt.title('Polynomial Regression Fit to y = x^2')  # 그래프의 제목을 설정합니다.
plt.xlabel('x')  # x축의 라벨을 설정합니다.
plt.ylabel('y')  # y축의 라벨을 설정합니다.
plt.legend()  # 범례를 표시합니다.
plt.grid(True)  # 그리드를 표시합니다.
plt.show()  # 그래프를 보여줍니다.

# 표로 나타내기
# 실제 값과 예측 값을 비교하는 작은 데이터 프레임 생성
df_comparison = pd.DataFrame({'x': x.flatten(), 'Actual y': y.flatten(), 'Predicted y': y_pred.flatten()})  # x, 실제 y, 예측 y를 포함하는 데이터 프레임을 생성합니다.
df_comparison = df_comparison.head(10)  # 처음 10개의 값을 표시합니다.
print(df_comparison)  # 데이터 프레임을 출력합니다.

