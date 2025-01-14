import pandas as pd
import numpy as np
import logging
import sys
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score
from datetime import datetime
import warnings
import time

warnings.filterwarnings('ignore')


# 로깅 설정
def setup_logging():
    log_dir = "model_logs/model_logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    log_file = os.path.join(
        log_dir,
        f"model_run_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
    )

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)


logger = setup_logging()


def preprocess_data(df, is_test=False):
    logger.info("데이터 전처리 시작")
    df = df.copy()

    # 이름에서 더 자세한 정보 추출
    df['LastName'] = df['Name'].str.split(',').str[0]
    df['Title'] = df['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)

    # Title 더 세분화
    df['Title'] = df['Title'].replace(
        ['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    df['Title'] = df['Title'].replace('Mlle', 'Miss')
    df['Title'] = df['Title'].replace(['Ms', 'Mme'], 'Mrs')

    # Cabin 정보 활용
    df['Deck'] = df['Cabin'].str[0]
    df['Deck'] = df['Deck'].fillna('Unknown')
    df['HasCabin'] = df['Cabin'].notna().astype(int)

    # Ticket 정보 활용
    df['TicketPrefix'] = df['Ticket'].str.extract('([A-Za-z]+)', expand=False)
    df['TicketPrefix'] = df['TicketPrefix'].fillna('NUM')
    df['TicketLen'] = df['Ticket'].str.len()

    # 가족 관련 특성 확장
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['IsAlone'] = (df['FamilySize'] == 1).astype(int)

    # Age 결측치 더 정교하게 처리
    age_by_pclass_sex = df.groupby(['Title', 'Pclass'])['Age'].transform('median')
    df['Age'] = df['Age'].fillna(age_by_pclass_sex)

    # Age 관련 특성 추가
    df['AgeBin'] = pd.qcut(df['Age'], 10, labels=False, duplicates='drop')
    df['IsChild'] = (df['Age'] < 16).astype(int)
    df['IsElderly'] = (df['Age'] > 60).astype(int)

    # Fare 관련 특성
    if is_test and df['Fare'].isnull().any():
        df['Fare'] = df['Fare'].fillna(df.groupby('Pclass')['Fare'].transform('median'))
    df['FareBin'] = pd.qcut(df['Fare'], 10, labels=False, duplicates='drop')
    df['FarePerPerson'] = df['Fare'] / df['FamilySize']

    # 상호작용 특성
    df['Age*Class'] = df['Age'] * df['Pclass']
    df['Fare*Class'] = df['Fare'] * df['Pclass']
    df['Age*Fare'] = df['Age'] * df['Fare']

    # Sex 인코딩
    df['Sex'] = (df['Sex'] == 'male').astype(int)

    # Embarked 처리
    df['Embarked'] = df['Embarked'].fillna('S')
    embarked_mapping = {'S': 0, 'C': 1, 'Q': 2}
    df['Embarked'] = df['Embarked'].map(embarked_mapping)

    # 범주형 변수 인코딩
    categorical_features = ['Title', 'Deck', 'TicketPrefix']
    for feat in categorical_features:
        df[feat] = LabelEncoder().fit_transform(df[feat].astype(str))

    # 최종 특성 선택
    features = ['Pclass', 'Sex', 'Age', 'Fare', 'Embarked', 'FamilySize',
                'IsAlone', 'Title', 'AgeBin', 'FareBin', 'HasCabin', 'Deck',
                'TicketPrefix', 'TicketLen', 'IsChild', 'IsElderly',
                'FarePerPerson', 'Age*Class', 'Fare*Class', 'Age*Fare']

    logger.info("데이터 전처리 완료")
    return df[features]


def optimize_weights(rf_pred, gb_pred, y_val):
    best_score = 0
    best_weights = (0.5, 0.5)

    for w1 in range(1, 10):
        w1 = w1 / 10
        w2 = 1 - w1
        weighted_pred = w1 * rf_pred + w2 * gb_pred
        pred = (weighted_pred > 0.5).astype(int)
        score = accuracy_score(y_val, pred)

        if score > best_score:
            best_score = score
            best_weights = (w1, w2)

    return best_weights


def main():
    try:
        logger.info("모델 학습 스크립트 시작")

        # 데이터 로드
        logger.info("데이터 로드 중...")
        train_data = pd.read_csv('train.csv')
        test_data = pd.read_csv('test.csv')
        logger.info("데이터 로드 완료")

        # 데이터 전처리
        logger.info("데이터 전처리 시작")
        start_time = time.time()
        X = preprocess_data(train_data)
        X_test = preprocess_data(test_data, is_test=True)
        y = train_data['Survived']

        # 학습 데이터와 검증 데이터 분리
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        end_time = time.time()
        logger.info(f"데이터 전처리 시간: {end_time - start_time:.2f}초")

        # 최종 결측치 처리
        logger.info("최종 결측치 처리 중...")
        imputer = SimpleImputer(strategy='median')
        X_train = pd.DataFrame(imputer.fit_transform(X_train), columns=X_train.columns)
        X_val = pd.DataFrame(imputer.transform(X_val), columns=X_val.columns)
        X_test = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns)

        # 그리드 서치 파라미터 설정
        logger.info("그리드 서치 파라미터 설정")
        rf_params = {
            'n_estimators': [800, 1000, 1200],
            'max_depth': [5, 6, 7, 8],
            'min_samples_split': [2, 3, 4],
            'min_samples_leaf': [1, 2, 3],
            'max_features': ['sqrt', 'log2', None]
        }

        gb_params = {
            'n_estimators': [800, 1000, 1200],
            'learning_rate': [0.01, 0.05, 0.1],
            'max_depth': [4, 5, 6],
            'min_samples_split': [2, 3, 4],
            'min_samples_leaf': [1, 2],
            'subsample': [0.8, 0.9, 1.0]
        }

        # 랜덤 포레스트 그리드 서치
        logger.info("랜덤 포레스트 그리드 서치 시작")
        rf_grid = GridSearchCV(
            RandomForestClassifier(random_state=42),
            rf_params,
            cv=5,
            scoring='accuracy',
            n_jobs=-1,
            verbose=2
        )
        rf_grid.fit(X_train, y_train)
        logger.info(f"Best Random Forest Parameters: {rf_grid.best_params_}")
        logger.info(f"Best RF CV Score: {rf_grid.best_score_}")

        # 그래디언트 부스팅 그리드 서치
        logger.info("그래디언트 부스팅 그리드 서치 시작")
        gb_grid = GridSearchCV(
            GradientBoostingClassifier(random_state=42),
            gb_params,
            cv=5,
            scoring='accuracy',
            n_jobs=-1,
            verbose=2
        )
        gb_grid.fit(X_train, y_train)
        logger.info(f"Best Gradient Boosting Parameters: {gb_grid.best_params_}")
        logger.info(f"Best GB CV Score: {gb_grid.best_score_}")

        # 검증 세트에서 최적의 가중치 찾기
        logger.info("최적 가중치 탐색 중...")
        rf_val_pred = rf_grid.predict_proba(X_val)[:, 1]
        gb_val_pred = gb_grid.predict_proba(X_val)[:, 1]
        best_weights = optimize_weights(rf_val_pred, gb_val_pred, y_val)
        logger.info(f"최적 가중치: RF={best_weights[0]:.2f}, GB={best_weights[1]:.2f}")

        # 테스트 세트 예측
        logger.info("테스트 세트 예측 중...")
        rf_pred = rf_grid.predict_proba(X_test)[:, 1]
        gb_pred = gb_grid.predict_proba(X_test)[:, 1]

        # 최종 예측
        final_pred = (rf_pred * best_weights[0] + gb_pred * best_weights[1])
        predictions = [1 if p > 0.5 else 0 for p in final_pred]

        # 제출 파일 생성
        logger.info("제출 파일 생성 중...")
        submission = pd.DataFrame({
            'PassengerId': test_data['PassengerId'],
            'Survived': predictions
        })
        submission.to_csv('submission.csv', index=False)
        logger.info("제출 파일 생성 완료")

        # 특성 중요도 출력
        logger.info("특성 중요도 분석 중...")
        feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'rf_importance': rf_grid.best_estimator_.feature_importances_,
            'gb_importance': gb_grid.best_estimator_.feature_importances_
        })
        feature_importance = feature_importance.sort_values('rf_importance', ascending=False)
        logger.info("\nFeature Importance:\n" + feature_importance.to_string())

        logger.info("모델 학습 스크립트 완료")

    except Exception as e:
        logger.error(f"오류 발생: {str(e)}", exc_info=True)


if __name__ == "__main__":
    main()