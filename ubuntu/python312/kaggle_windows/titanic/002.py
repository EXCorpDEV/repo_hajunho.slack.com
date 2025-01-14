import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
from sklearn.metrics import accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

# 데이터 로드
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')


def preprocess_data(df):
    # 이름에서 직함 추출 (escape sequence 수정)
    df['Title'] = df['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)

    # 희귀한 직함을 'Rare'로 통합
    rare_titles = ['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona']
    df['Title'] = df['Title'].replace('Mlle', 'Miss')
    df['Title'] = df['Title'].replace('Ms', 'Miss')
    df['Title'] = df['Title'].replace('Mme', 'Mrs')
    df['Title'] = df['Title'].replace(rare_titles, 'Rare')

    # Age 결측치 처리 - Title 기반으로 대체
    age_map = df.groupby('Title')['Age'].median()
    for title in age_map.index:
        df.loc[(df['Age'].isnull()) & (df['Title'] == title), 'Age'] = age_map[title]

    # Fare 결측치 처리
    df['Fare'] = df['Fare'].fillna(df['Fare'].median())

    # Embarked 결측치 처리
    df['Embarked'] = df['Embarked'].fillna('S')

    # 가족 관련 특성
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['IsAlone'] = (df['FamilySize'] == 1).astype(int)

    # Cabin 정보 활용
    df['HasCabin'] = df['Cabin'].notna().astype(int)

    # 범주형 변수 인코딩
    categorical_features = ['Sex', 'Embarked', 'Title']
    for feature in categorical_features:
        le = LabelEncoder()
        df[feature] = le.fit_transform(df[feature].astype(str))

    # 연속형 변수 구간화
    df['Age_Band'] = pd.cut(df['Age'], 5, labels=False)
    df['Fare_Band'] = pd.qcut(df['Fare'], 5, labels=False)

    # 상호작용 특성
    df['Age*Class'] = df['Age'] * df['Pclass']
    df['Fare*Class'] = df['Fare'] * df['Pclass']

    features = ['Pclass', 'Sex', 'Age', 'Fare', 'Embarked', 'FamilySize', 'IsAlone',
                'HasCabin', 'Title', 'Age_Band', 'Fare_Band', 'Age*Class', 'Fare*Class']

    return df[features]


# 전처리 적용
X_train = preprocess_data(train_data)
X_test = preprocess_data(test_data)
y_train = train_data['Survived']

# DMatrix 형식으로 변환
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test)

# XGBoost 파라미터 설정
params = {
    'max_depth': 4,
    'learning_rate': 0.01,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'seed': 42
}

# 모델 학습
num_rounds = 1000
model = xgb.train(params, dtrain, num_rounds)

# 예측
predictions = model.predict(dtest)
predictions = [1 if pred > 0.5 else 0 for pred in predictions]

# 제출 파일 생성
submission = pd.DataFrame({
    'PassengerId': test_data['PassengerId'],
    'Survived': predictions
})
submission.to_csv('submission.csv', index=False)

# 특성 중요도 시각화
importance = model.get_score(importance_type='weight')
importance = pd.DataFrame.from_dict(importance, orient='index', columns=['importance'])
importance = importance.sort_values('importance', ascending=False)

plt.figure(figsize=(12, 8))
sns.barplot(x='importance', y=importance.index, data=importance)
plt.title('Feature Importance')
plt.tight_layout()
plt.show()