import pandas as pd
import numpy as np
import logging
import sys
import os
import subprocess
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from datetime import datetime
import warnings
import time

warnings.filterwarnings('ignore')


# 로깅 설정
def setup_logging():
    log_dir = "model_logs"
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
    logger.info("데이터 전처리 시작.")
    df = df.copy()

    # Title 추출 및 매핑
    logger.debug("Title 추출 및 매핑 중...")
    df['Title'] = df['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
    title_mapping = {
        "Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4,
        "Dr": 5, "Rev": 5, "Col": 5, "Major": 5, "Mlle": 2,
        "Countess": 5, "Ms": 2, "Lady": 5, "Jonkheer": 5,
        "Don": 5, "Mme": 3, "Capt": 5, "Sir": 5
    }
    df['Title'] = df['Title'].map(title_mapping).fillna(0)

    # 가족 관련 특성
    logger.debug("가족 관련 특성 생성 중...")
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['IsAlone'] = (df['FamilySize'] == 1).astype(int)

    # Fare 처리
    logger.debug("Fare 처리 중...")
    if is_test and df['Fare'].isnull().any():
        df['Fare'] = df['Fare'].fillna(df.groupby('Pclass')['Fare'].transform('median'))

    # Sex 인코딩
    logger.debug("Sex 인코딩 중...")
    df['Sex'] = (df['Sex'] == 'male').astype(int)

    # Embarked 처리
    logger.debug("Embarked 처리 중...")
    df['Embarked'] = df['Embarked'].fillna('S')
    embarked_mapping = {'S': 0, 'C': 1, 'Q': 2}
    df['Embarked'] = df['Embarked'].map(embarked_mapping)

    # Age 처리
    logger.debug("Age 처리 중...")
    df['Age'] = df['Age'].fillna(df.groupby('Title')['Age'].transform('median'))
    df['AgeBin'] = pd.cut(df['Age'], bins=[0, 12, 20, 30, 50, 80], labels=[0, 1, 2, 3, 4])

    # Fare 구간화
    logger.debug("Fare 구간화 중...")
    df['FareBin'] = pd.qcut(df['Fare'], 6, labels=False)

    # 특성 선택
    features = ['Pclass', 'Sex', 'Age', 'Fare', 'Embarked', 'FamilySize',
                'IsAlone', 'Title', 'AgeBin', 'FareBin']
    logger.info("데이터 전처리 완료.")
    return df[features]


def main():
    try:
        logger.info("모델 학습 스크립트 시작.")

        # 데이터 로드
        logger.info("데이터 로드 중...")
        train_data = pd.read_csv('train.csv')
        test_data = pd.read_csv('test.csv')
        logger.info("데이터 로드 완료.")

        # 데이터 전처리 타이밍
        logger.info("데이터 전처리 시작.")
        start_time = time.time()
        X_train = preprocess_data(train_data)
        X_test = preprocess_data(test_data, is_test=True)
        y_train = train_data['Survived']
        end_time = time.time()
        logger.info(f"데이터 전처리 시간: {end_time - start_time:.2f}초")

        # 최종 결측치 처리
        logger.info("최종 결측치 처리 중...")
        start_time = time.time()
        imputer = SimpleImputer(strategy='median')
        X_train = pd.DataFrame(imputer.fit_transform(X_train), columns=X_train.columns)
        X_test = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns)
        end_time = time.time()
        logger.info(f"최종 결측치 처리 시간: {end_time - start_time:.2f}초")

        # 특성 선택 (필요 시)
        # selected_features = ['Pclass', 'Sex', 'Age', 'Fare', 'Embarked', 'FamilySize', 'IsAlone', 'Title']
        # X_train = X_train[selected_features]
        # X_test = X_test[selected_features]

        # 그리드 서치 파라미터 설정
        logger.info("그리드 서치 파라미터 설정 중...")
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
        logger.info("그리드 서치 파라미터 설정 완료.")

        # 랜덤 포레스트 그리드 서치
        logger.info("랜덤 포레스트 그리드 서치 시작.")
        rf_grid = GridSearchCV(
            RandomForestClassifier(random_state=42),
            rf_params,
            cv=5,
            scoring='accuracy',
            n_jobs=-1,
            verbose=3
        )
        start_time = time.time()
        rf_grid.fit(X_train, y_train)
        end_time = time.time()
        logger.info(f"랜덤 포레스트 그리드 서치 완료. 소요 시간: {end_time - start_time:.2f}초")

        # 그라디언트 부스팅 그리드 서치
        logger.info("그라디언트 부스팅 그리드 서치 시작.")
        gb_grid = GridSearchCV(
            GradientBoostingClassifier(random_state=42),
            gb_params,
            cv=5,
            scoring='accuracy',
            n_jobs=-1,
            verbose=3
        )
        start_time = time.time()
        gb_grid.fit(X_train, y_train)
        end_time = time.time()
        logger.info(f"그라디언트 부스팅 그리드 서치 완료. 소요 시간: {end_time - start_time:.2f}초")

        # 최적 파라미터 및 스코어 출력
        logger.info("최적 파라미터 및 스코어 출력 중...")
        logger.info(f"Best Random Forest Parameters: {rf_grid.best_params_}")
        logger.info(f"Best RF CV Score: {rf_grid.best_score_}")
        logger.info(f"Best Gradient Boosting Parameters: {gb_grid.best_params_}")
        logger.info(f"Best GB CV Score: {gb_grid.best_score_}")

        # 최적화된 모델로 예측
        logger.info("최적화된 모델로 예측 중...")
        rf_pred = rf_grid.predict_proba(X_test)[:, 1]
        gb_pred = gb_grid.predict_proba(X_test)[:, 1]

        # 앙상블 예측 (동일 가중치)
        logger.info("앙상블 예측 수행 중...")
        final_pred = (rf_pred * 0.5 + gb_pred * 0.5)
        predictions = [1 if p > 0.5 else 0 for p in final_pred]

        # 제출 파일 생성
        logger.info("제출 파일 생성 중...")
        submission = pd.DataFrame({
            'PassengerId': test_data['PassengerId'],
            'Survived': predictions
        })
        submission.to_csv('submission.csv', index=False)
        logger.info("제출 파일 'submission.csv' 생성 완료.")

        # 특성 중요도 출력
        logger.info("특성 중요도 출력 중...")
        feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'rf_importance': rf_grid.best_estimator_.feature_importances_,
            'gb_importance': gb_grid.best_estimator_.feature_importances_
        })
        feature_importance = feature_importance.sort_values('rf_importance', ascending=False)
        logger.info("\nFeature Importance:")
        logger.info(feature_importance.to_string(index=False))

        logger.info("모델 학습 스크립트 완료.")

    except Exception as e:
        logger.error(f"예외 발생: {str(e)}", exc_info=True)


if __name__ == "__main__":
    main()
