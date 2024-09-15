from typing import Dict, List
import numpy as np
import pandas as pd

def linear_interpolation(time_series_data: pd.DataFrame) -> pd.DataFrame:
    """
    선형 보간을 사용하여 시계열 데이터를 보간하는 함수

    Args:
        time_series_data (pd.DataFrame): 시계열 데이터프레임 (shape: [n_timesteps, n_features])
            - n_timesteps: 시계열 데이터의 시간 스텝 수 (행의 개수)
            - n_features: 변수 또는 특성(feature)의 수 (열의 개수)

    Returns:
        pd.DataFrame: 보간된 데이터프레임 (shape: [n_timesteps, n_features])
            - n_timesteps: 보간 후의 시간 스텝 수 (행의 개수)
            - n_features: 변수 또는 특성(feature)의 수 (열의 개수)
    """
    # 선형 보간을 사용하여 결측값(NaN)을 채움 (shape: [n_timesteps, n_features])
    interpolated_data = time_series_data.interpolate(method='linear', axis=0)
    
    # 선형 보간 후, 남아있는 결측값을 채우기 위해 역방향 채우기(bfill)와 정방향 채우기(ffill)를 적용 (shape: [n_timesteps, n_features])
    interpolated_data = interpolated_data.bfill().ffill()
    
    # 최종적으로 보간이 완료된 데이터프레임을 반환 (shape: [n_timesteps, n_features])
    return interpolated_data
