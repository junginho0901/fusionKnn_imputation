from typing import Dict, List
import numpy as np
import pandas as pd

def cubicspline_interpolation(time_series_data: pd.DataFrame) -> pd.DataFrame:
    """
    Cubic Spline 보간법을 사용하여 결측 값을 채우는 함수

    Args:
        time_series_data (pd.DataFrame): 보간할 시계열 데이터프레임 (shape: [n_timesteps, n_features])
            - n_timesteps: 시간의 흐름에 따라 기록된 데이터 포인트의 수 (행의 개수)
            - n_features: 데이터의 특성의(feature) 수 (열의 개수)

    Returns:
        pd.DataFrame: 결측 값이 Cubic Spline 보간법으로 채워진 데이터프레임 (shape: [n_timesteps, n_features])
            - n_timesteps: 보간된 데이터의 시간의 흐름에 따른 데이터 포인트 수 (행의 개수)
            - n_features: 데이터의 특성의(feature) 수 (열의 개수)
    """
    # Cubic Spline 보간을 사용하여 결측값(NaN)을 채움. 각 열(column)에 대해 독립적으로 처리 (shape: [n_timesteps, n_features])
    interpolated_data = time_series_data.interpolate(method='cubicspline')
    
    # 최종적으로 Cubic Spline 보간법을 사용해 결측값이 채워진 데이터프레임 반환 (shape: [n_timesteps, n_features])
    return interpolated_data
