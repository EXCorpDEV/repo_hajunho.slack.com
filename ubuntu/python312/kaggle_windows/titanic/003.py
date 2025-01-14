import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

# 데이터 로드
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')


def preprocess_data(df):
    # 기본 전처리
    df['Title'] = df['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)

    # Title 그룹화 개선
    df['Title'] = df['Title'].replace(
        ['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    df['Title'] = df['Title'].replace(['Mlle', 'Ms', 'Mme'], 'Miss')

    # Age 결측치 더 정교하게 처리
    age_by_class_sex = df.groupby(['Title', 'Pclass'])['Age'].median()
    for (title, pclass), age in age_by_class_sex.items():
        df.loc[(df['Age'].isnull()) & (df['Title'] == title) & (df['Pclass'] == pclass), 'Age'] = age

    # 결측치 처리
    df['Age'] = df['Age'].fillna(df['Age'].median())
    df['Fare'] = df['Fare'].fillna(df['Fare'].median())
    df['Embarked'] = df['Embarked'].fillna('S')

    # 가족 관련 특성
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['IsAlone'] = (df['FamilySize'] == 1).astype(int)

    # 새로운 특성
    df['Age*Pclass'] = df['Age'] * df['Pclass']
    df['Fare_Per_Person'] = df['Fare'] / df['FamilySize']

    # 구간화
    df['AgeBin'] = pd.qcut(df['Age'], 8, labels=False, duplicates='drop')
    df['FareBin'] = pd.qcut(df['Fare'], 8, labels=False, duplicates='drop')

    # 범주형 변수 인코딩
    for column in ['Sex', 'Embarked', 'Title']:
        df[column] = LabelEncoder().fit_transform(df[column].astype(str))

    features = ['Pclass', 'Sex', 'Age', 'Fare', 'Embarked', 'FamilySize', 'IsAlone',
                'Title', 'AgeBin', 'FareBin', 'Age*Pclass', 'Fare_Per_Person']
    return df[features]


# 데이터 전처리
X_train = preprocess_data(train_data)
X_test = preprocess_data(test_data)
y_train = train_data['Survived']

# XGBoost 모델
xgb_params = {
    'max_depth': 4,
    'learning_rate': 0.01,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'seed': 42
}
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test)
xgb_model = xgb.train(xgb_params, dtrain, num_boost_round=1000)
xgb_pred = xgb_model.predict(dtest)

# Random Forest 모델
rf_model = RandomForestClassifier(
    n_estimators=1000,
    max_depth=5,
    min_samples_split=4,
    min_samples_leaf=2,
    random_state=42
)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict_proba(X_test)[:, 1]

# LightGBM 모델
lgb_model = LGBMClassifier(
    n_estimators=1000,
    learning_rate=0.01,
    max_depth=4,
    num_leaves=31,
    random_state=42
)
lgb_model.fit(X_train, y_train)
lgb_pred = lgb_model.predict_proba(X_test)[:, 1]

# 앙상블 예측 (가중 평균)
final_pred = (xgb_pred * 0.4 + rf_pred * 0.3 + lgb_pred * 0.3)
predictions = [1 if pred > 0.5 else 0 for pred in final_pred]

# 제출 파일 생성
submission = pd.DataFrame({
    'PassengerId': test_data['PassengerId'],
    'Survived': predictions
})
submission.to_csv('submission.csv', index=False)