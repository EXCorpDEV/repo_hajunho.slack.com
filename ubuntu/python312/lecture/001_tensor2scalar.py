import numpy as np

# 2차원 텐서 생성
tensor_2d = np.array([[1, 2, 3], [4, 5, 6]])

# 합계 계산
scalar_sum = np.sum(tensor_2d)
print("합계:", scalar_sum)  # 출력: 합계: 21

# 평균 계산
scalar_mean = np.mean(tensor_2d)
print("평균:", scalar_mean)  # 출력: 평균: 3.5
