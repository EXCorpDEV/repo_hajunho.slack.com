import torch
import torch.nn as nn
import torch.optim as optim

# 데이터 생성
x_train = torch.linspace(-10, 10, 100).view(-1, 1) # x 값
y_train = 2 * x_train # y = 2x

# 모델 정의
class LinearRegressionModel(nn.Module):
    def __init__(self):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(in_features=1, out_features=1) # 입력 차원 1, 출력 차원 1

    def forward(self, x):
        return self.linear(x)

# 모델 인스턴스 생성
model = LinearRegressionModel()

# 손실 함수 및 옵티마이저 정의
criterion = nn.MSELoss() # 평균 제곱 오차 손실
optimizer = optim.SGD(model.parameters(), lr=0.01) # 확률적 경사 하강법

# 학습 과정
epochs = 100
for epoch in range(epochs):
    # 예측
    y_pred = model(x_train)

    # 손실 계산
    loss = criterion(y_pred, y_train)

    # 기울기 초기화
    optimizer.zero_grad()
    # 손실에 대한 기울기 계산
    loss.backward()
    # 파라미터 업데이트
    optimizer.step()

    # 학습 과정 출력
    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

# 학습된 모델의 가중치와 편향 출력
trained_weight = model.linear.weight.item()
trained_bias = model.linear.bias.item()

print(f'Results = {trained_weight}, {trained_bias}')

