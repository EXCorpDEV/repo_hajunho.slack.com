import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.impute import SimpleImputer
import warnings

warnings.filterwarnings('ignore')

# 데이터 로드
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')


def preprocess_data(df, is_test=False):
    # 복사본 생성
    df = df.copy()

    # 기본 특성 추출
    df['Title'] = df['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)

    # Title 매핑
    title_mapping = {
        "Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4,
        "Dr": 5, "Rev": 5, "Col": 5, "Major": 5, "Mlle": 2,
        "Countess": 5, "Ms": 2, "Lady": 5, "Jonkheer": 5,
        "Don": 5, "Mme": 3, "Capt": 5, "Sir": 5
    }
    df['Title'] = df['Title'].map(title_mapping).fillna(0)

    # 가족 관련 특성
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['IsAlone'] = (df['FamilySize'] == 1).astype(int)

    # Fare 처리
    if is_test and df['Fare'].isnull().any():
        df['Fare'] = df['Fare'].fillna(df.groupby('Pclass')['Fare'].transform('median'))

    # Sex 인코딩
    df['Sex'] = (df['Sex'] == 'male').astype(int)

    # Embarked 처리
    df['Embarked'] = df['Embarked'].fillna('S')
    embarked_mapping = {'S': 0, 'C': 1, 'Q': 2}
    df['Embarked'] = df['Embarked'].map(embarked_mapping)

    # 나이 구간화
    df['Age'] = df['Age'].fillna(df.groupby('Title')['Age'].transform('median'))
    df['AgeBin'] = pd.cut(df['Age'], bins=[0, 12, 20, 30, 50, 80], labels=[0, 1, 2, 3, 4])

    # Fare 구간화
    df['FareBin'] = pd.qcut(df['Fare'], 6, labels=False)

    # 선택할 특성들
    features = ['Pclass', 'Sex', 'Age', 'Fare', 'Embarked', 'FamilySize',
                'IsAlone', 'Title', 'AgeBin', 'FareBin']

    return df[features]


# 데이터 전처리
X_train = preprocess_data(train_data)
X_test = preprocess_data(test_data, is_test=True)
y_train = train_data['Survived']

# 최종 결측치 처리
imputer = SimpleImputer(strategy='median')
X_train = pd.DataFrame(imputer.fit_transform(X_train), columns=X_train.columns)
X_test = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns)

# 모델 정의
rf_model = RandomForestClassifier(
    n_estimators=1000,
    max_depth=5,
    min_samples_split=4,
    min_samples_leaf=2,
    random_state=42
)

gb_model = GradientBoostingClassifier(
    n_estimators=1000,
    learning_rate=0.01,
    max_depth=4,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42
)

xgb_model = XGBClassifier(
    n_estimators=1000,
    learning_rate=0.01,
    max_depth=4,
    min_child_weight=2,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

# 모델 학습
rf_model.fit(X_train, y_train)
gb_model.fit(X_train, y_train)
xgb_model.fit(X_train, y_train)

# 예측
rf_pred = rf_model.predict_proba(X_test)[:, 1]
gb_pred = gb_model.predict_proba(X_test)[:, 1]
xgb_pred = xgb_model.predict_proba(X_test)[:, 1]

# 앙상블 예측 (가중 평균)
final_pred = (rf_pred * 0.3 + gb_pred * 0.3 + xgb_pred * 0.4)
predictions = [1 if p > 0.5 else 0 for p in final_pred]

# 제출 파일 생성
submission = pd.DataFrame({
    'PassengerId': test_data['PassengerId'],
    'Survived': predictions
})
submission.to_csv('submission.csv', index=False)

# 특성 중요도 출력
feature_importance = pd.DataFrame({
    'feature': X_train.columns,
    'importance': rf_model.feature_importances_
})
print("\nFeature Importance:")
print(feature_importance.sort_values('importance', ascending=False))