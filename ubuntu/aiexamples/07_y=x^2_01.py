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

# 모델 파라미터
coefficients = poly_reg_model.coef_
intercept = poly_reg_model.intercept_

# 학습된 모델로부터 도출된 수식 확인
print(coefficients, intercept)
