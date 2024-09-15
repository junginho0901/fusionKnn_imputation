from typing import Dict, List
import numpy as np
import pandas as pd

def moving_average_interpolation(time_series_data: pd.DataFrame, window: int) -> pd.DataFrame:
    """
    Moving average 보간을 사용하여 시계열 데이터를 보간하는 함수

    Args:
        time_series_data (pd.DataFrame): 시계열 데이터프레임 (shape: [n_timesteps, n_features])
        window (int): 이동 평균을 계산할 윈도우 크기

    Returns:
        pd.DataFrame: 보간된 데이터프레임 (shape: [n_timesteps, n_features])
    """
    # 입력 데이터프레임을 복사하여 작업 (shape: [n_timesteps, n_features])
    interpolated_data = time_series_data.copy()
    
    # 각 열(column)에 대해 이동 평균 보간 적용
    for column in interpolated_data.columns:
        # 이동 평균을 사용하여 NaN 값을 보간 (shape: [n_timesteps])
        interpolated_data[column] = interpolated_data[column].fillna(
            interpolated_data[column].rolling(window=window, min_periods=1, center=True).mean()
        )
    
    # 남아있는 NaN 값을 채우기 위해 forward fill과 backward fill 적용 (shape: [n_timesteps, n_features])
    interpolated_data = interpolated_data.ffill().bfill()
    
    # 최종적으로 보간된 데이터프레임 반환 (shape: [n_timesteps, n_features])
    return interpolated_data
