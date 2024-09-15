from typing import Dict, List
import numpy as np
import pandas as pd

def nearest_interpolation(time_series_data: pd.DataFrame) -> pd.DataFrame:
    """
    Nearest interpolation을 사용하여 시계열 데이터를 보간하는 함수

    Args:
        time_series_data (pd.DataFrame): 시계열 데이터프레임 (shape: [n_timesteps, n_features])

    Returns:
        pd.DataFrame: 보간된 데이터프레임 (shape: [n_timesteps, n_features])
    """
    # Nearest interpolation을 사용하여 결측값(NaN)을 채움 (shape: [n_timesteps, n_features])
    interpolated_data = time_series_data.interpolate(method='nearest', axis=0)
    
    # 최종적으로 모든 결측값이 채워진 데이터프레임 반환 (shape: [n_timesteps, n_features])
    return interpolated_data
