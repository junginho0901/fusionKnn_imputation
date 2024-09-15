from typing import Dict, List
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer

def knn_imputation(time_series_data: pd.DataFrame, n_neighbors: int) -> pd.DataFrame:
    """
    KNN을 사용하여 시계열 데이터를 보간하는 함수

    Args:
        time_series_data (pd.DataFrame): 시계열 데이터프레임 (shape: [n_timesteps, n_features])
            - n_timesteps: 타임스텝의 수 (행의 개수)
            - n_features: 변수 또는 특성(feature)의 수 (열의 개수)
        n_neighbors (int): KNN의 이웃 수

    Returns:
        pd.DataFrame: 보간된 데이터프레임 (shape: [n_timesteps, n_features])
            - n_timesteps: 타임스텝의 수 (행의 개수)
            - n_features: 변수 또는 특성(feature)의 수 (열의 개수)
    """
    imputer = KNNImputer(n_neighbors=n_neighbors)  # KNN 보간기를 설정 (n_neighbors로 설정된 이웃 수 사용)
    # time_series_data에 대해 KNN 보간을 수행하고, 결과를 새로운 데이터프레임으로 반환 (shape: [n_timesteps, n_features])
    interpolated_data = pd.DataFrame(
        imputer.fit_transform(time_series_data),  # KNN 보간 실행 (shape: [n_timesteps, n_features])
        columns=time_series_data.columns,  # 원본 데이터의 열 이름 유지
        index=time_series_data.index  # 원본 데이터의 인덱스 유지
    )
    return interpolated_data  # 보간된 데이터프레임 반환
