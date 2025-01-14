import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV
import warnings

warnings.filterwarnings('ignore')

# 데이터 로드
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')


def preprocess_data(df, is_test=False):
    df = df.copy()

    # Title 추출 및 매핑
    df['Title'] = df['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
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

    # Age 처리
    df['Age'] = df['Age'].fillna(df.groupby('Title')['Age'].transform('median'))
    df['AgeBin'] = pd.cut(df['Age'], bins=[0, 12, 20, 30, 50, 80], labels=[0, 1, 2, 3, 4])

    # Fare 구간화
    df['FareBin'] = pd.qcut(df['Fare'], 6, labels=False)

    # 특성 선택
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

# Grid Search 파라미터 설정

# Grid Search 파라미터 설정 - XGBoost 제외
rf_params = {
    'n_estimators': [500, 800, 1000],
    'max_depth': [4, 5, 6],
    'min_samples_split': [2, 4, 6],
    'min_samples_leaf': [1, 2, 4]
}

gb_params = {
    'n_estimators': [500, 800, 1000],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 4, 5],
    'min_samples_split': [2, 4],
    'min_samples_leaf': [1, 2]
}

# Grid Search 수행
print("Starting Random Forest Grid Search...")
rf_grid = GridSearchCV(RandomForestClassifier(random_state=42),
                      rf_params, cv=5, scoring='accuracy', n_jobs=-1)
rf_grid.fit(X_train, y_train)

print("\nStarting Gradient Boosting Grid Search...")
gb_grid = GridSearchCV(GradientBoostingClassifier(random_state=42),
                      gb_params, cv=5, scoring='accuracy', n_jobs=-1)
gb_grid.fit(X_train, y_train)

# 최적 파라미터 출력
print("\nBest Random Forest Parameters:", rf_grid.best_params_)
print("Best RF CV Score:", rf_grid.best_score_)
print("\nBest Gradient Boosting Parameters:", gb_grid.best_params_)
print("Best GB CV Score:", gb_grid.best_score_)

# 최적화된 모델로 예측
rf_pred = rf_grid.predict_proba(X_test)[:, 1]
gb_pred = gb_grid.predict_proba(X_test)[:, 1]

# 앙상블 예측 (동일 가중치)
final_pred = (rf_pred * 0.5 + gb_pred * 0.5)
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
    'rf_importance': rf_grid.best_estimator_.feature_importances_,
    'gb_importance': gb_grid.best_estimator_.feature_importances_
})

print("\nFeature Importance:")
print(feature_importance.sort_values('rf_importance', ascending=False))