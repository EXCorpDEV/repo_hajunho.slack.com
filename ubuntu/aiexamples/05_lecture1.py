import matplotlib.pyplot as plt
import numpy as np

# 딸의 나이와 키 데이터를 입력합니다.
# 예를 들어, (나이, 키) 데이터
age = np.array([5, 6, 7, 8, 9, 10, 11, 12, 13])
height = np.array([105, 112, 119, 125, 132, 138, 142, 146, 148])  # 대략적인 데이터로, 실제 데이터를 입력하면 됩니다.

# 단순 선형 회귀 모델을 사용하여 방정식을 찾습니다.
# y = ax + b 형태의 방정식을 찾습니다.
a, b = np.polyfit(age, height, 1)  # 1차 방정식을 찾습니다.

# 예측값 계산
age_future = np.linspace(5, 18, 100)  # 5세부터 18세까지 나이를 설정합니다.
height_pred = a * age_future + b  # 예측된 키를 계산합니다.

# 그래프를 그립니다.
plt.figure(figsize=(10, 6))
plt.scatter(age, height, color='blue', label='Actual Height')
plt.plot(age_future, height_pred, color='red', label=f'Predicted Height: y = {a:.2f}x + {b:.2f}')
plt.title('Height Growth Over Time')
plt.xlabel('Age (years)')
plt.ylabel('Height (cm)')
plt.legend()
plt.grid(True)
plt.show()

# 방정식과 예측을 설명합니다.
print(f"딸의 키 성장 방정식은 y = {a:.2f}x + {b:.2f} 입니다.")
print("이 방정식은 나이가 x일 때 키 y를 예측합니다.")
print("예를 들어, 나이가 14세일 때 예상 키는 다음과 같습니다.")
age_14 = 14
predicted_height_14 = a * age_14 + b
print(f"예상 키: {predicted_height_14:.2f} cm")

# 데이터 프레임으로 실제 값과 예측 값을 비교하는 표를 생성합니다.
import pandas as pd
df_comparison = pd.DataFrame({'Age': age, 'Actual Height': height})
df_comparison['Predicted Height'] = a * df_comparison['Age'] + b
print(df_comparison)
