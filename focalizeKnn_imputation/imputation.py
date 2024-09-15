from typing import Dict, List
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer

def knn_impute_column(data: pd.DataFrame, column: str, selected_features: List[str], n_neighbors: int) -> np.ndarray:
    """
    KNN을 사용하여 특정 열을 보간하는 함수

    Args:
        data (pd.DataFrame): 데이터프레임 (shape: [n_timesteps, n_features])
        column (str): 보간할 열의 이름
        selected_features (List[str]): 선택된 특징 리스트 (shape: [n_selected_features])
        n_neighbors (int): KNN 이웃의 수

    Returns:
        np.ndarray: 보간된 값 배열 (shape: [n_timesteps])
    """
    print(f"Starting KNN imputation for column: {column}")
    imputation_data = data[selected_features]  # 보간에 사용할 데이터 서브셋 (shape: [n_timesteps, n_selected_features])
    imputer = KNNImputer(n_neighbors=n_neighbors, weights='uniform')  # KNN 보간기 생성
    imputed_values = imputer.fit_transform(imputation_data)  # 보간 수행 (shape: [n_timesteps, n_selected_features])
    print(f"KNN imputation completed for column: {column}")
    return imputed_values[:, 0]  # 보간된 첫 번째 열의 값을 반환 (shape: [n_timesteps])

def focalize_knn_imputation(data: pd.DataFrame, n_neighbors: int, threshold: float, lags: List[int]) -> pd.DataFrame:
    """
    Focalize KNN을 사용하여 데이터프레임을 보간하는 함수

    Args:
        data (pd.DataFrame): 결측값이 포함된 데이터프레임 (shape: [n_timesteps, n_features])
        n_neighbors (int): KNN 이웃의 수
        threshold (float): 상관계수 임계값
        lags (List[int]): 시차 리스트(도메인 지식에 따라 지정, 예: [24, 168])

    Returns:
        pd.DataFrame: 결측 값이 보간된 데이터프레임 (shape: [n_timesteps, n_features])
    """
    print("Starting focalize KNN imputation...")
    imputed_data = data.copy()  # 원본 데이터의 복사본 (shape: [n_timesteps, n_features])
    
    columns_with_nan = [col for col in data.columns if data[col].isna().any()]  # 결측값이 있는 열 리스트 (shape: [n_columns_with_nan])
    print(f"Columns with missing values: {columns_with_nan}")
    
    for column in columns_with_nan:
        print(f"\nProcessing column: {column}")
        correlations = data.corr()[column].abs()  # 모든 열과의 상관계수 계산 (shape: [n_features])
        selected_features = correlations[correlations > threshold].index.tolist()  # 임계값을 초과하는 특징 선택 (shape: [n_selected_features])
        print(f"Selected features for {column}: {selected_features}")
        
        for lag in lags:
            lagged_feature = data[column].shift(lag)  # 시차(lag)를 적용한 데이터 생성 (shape: [n_timesteps])
            lagged_feature_name = f"{column}_lag_{lag}"
            selected_features.append(lagged_feature_name)  # 시차가 적용된 열 이름 추가
            data[lagged_feature_name] = lagged_feature  # 시차 적용 데이터를 데이터프레임에 추가 (shape: [n_timesteps, n_features + n_lags])
        print(f"Added lagged features for {column}: {[f'{column}_lag_{lag}' for lag in lags]}")
        
        imputed_values = knn_impute_column(data, column, selected_features, n_neighbors)  # KNN 보간 수행 (shape: [n_timesteps])
        imputed_data[column] = imputed_values  # 보간된 값을 데이터프레임에 저장 (shape: [n_timesteps, n_features])
        print(f"Imputation completed for column: {column}")
    
    columns_to_drop = [col for col in imputed_data.columns if '_lag_' in col]  # 시차 적용된 열 제거 (shape: [n_lagged_columns])
    imputed_data = imputed_data.drop(columns=columns_to_drop)  # 시차 열 제거 후 최종 데이터프레임 (shape: [n_timesteps, n_features])
    print(f"Removed {len(columns_to_drop)} lagged columns")
    
    print("Focalize KNN imputation completed.")
    return imputed_data  # 최종 보간된 데이터프레임 반환 (shape: [n_timesteps, n_features])
