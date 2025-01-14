import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

# 데이터 로드
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

# 기본적인 데이터 탐색
print("Train Data Shape:", train_data.shape)
print("\nMissing Values in Train Data:")
print(train_data.isnull().sum())


# 데이터 전처리 함수
def preprocess_data(df):
    # Age 결측치 처리 - 중앙값으로 대체
    df['Age'] = df['Age'].fillna(df['Age'].median())

    # Embarked 결측치 처리 - 최빈값으로 대체
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

    # Cabin은 결측치가 많아서 있음/없음으로만 분류
    df['Cabin'] = df['Cabin'].apply(lambda x: 1 if isinstance(x, str) else 0)

    # Sex를 숫자로 변환
    df['Sex'] = LabelEncoder().fit_transform(df['Sex'])

    # Embarked를 숫자로 변환
    df['Embarked'] = LabelEncoder().fit_transform(df['Embarked'])

    # 가족 크기 특성 추가
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1

    # 사용할 특성 선택
    features = ['Pclass', 'Sex', 'Age', 'Fare', 'Embarked', 'FamilySize', 'Cabin']
    return df[features]


# 전처리 적용
X_train = preprocess_data(train_data)
X_test = preprocess_data(test_data)
y_train = train_data['Survived']

# 모델 학습
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 예측
predictions = model.predict(X_test)

# 제출 파일 생성
submission = pd.DataFrame({
    'PassengerId': test_data['PassengerId'],
    'Survived': predictions
})
submission.to_csv('submission.csv', index=False)

# 특성 중요도 시각화
feature_importance = pd.DataFrame({
    'feature': X_train.columns,
    'importance': model.feature_importances_
})
feature_importance = feature_importance.sort_values('importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='importance', y='feature', data=feature_importance)
plt.title('Feature Importance')
plt.show()