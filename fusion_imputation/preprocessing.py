import pandas as pd
import pandas as pd
from typing import List, Tuple
from fusion_imputation.utils import find_columns_with_nan, get_columns_with_nan

def interpolate_linear(time_series_data: pd.DataFrame) -> pd.DataFrame:
    """
    선형 보간법을 사용하여 결측 값을 채우는 함수

    Args:
        time_series_data (pd.DataFrame): 보간할 시계열 데이터프레임

    Returns:
        pd.DataFrame: 결측 값이 선형으로 채워진 데이터프레임
    """
    # NaN 값을 포함하는 열들을 찾음
    columns_with_nan = get_columns_with_nan(time_series_data)
    
    # 원본 데이터의 복사본을 생성하여 원본 데이터 보존
    interpolated_data = time_series_data.copy()
    
    # NaN을 포함하는 열들에 대해서만 선형 보간 적용
    # list()로 감싸는 이유는 pandas가 리스트 형태의 열 이름을 선호하기 때문
    # interpolate(): 선형 보간 적용
    # bfill(): 데이터의 시작 부분의 NaN 값들을 뒤의 값으로 채움
    # ffill(): 데이터의 끝 부분의 NaN 값들을 앞의 값으로 채움
    interpolated_data[list(columns_with_nan)] = interpolated_data[list(columns_with_nan)].interpolate(method='linear', axis=0).bfill().ffill()
    
    # 결측치가 채워진 데이터프레임 반환
    return interpolated_data

def rescale_to_original(original_data: pd.DataFrame, filled_data: pd.DataFrame) -> pd.DataFrame:
    """
    보간된 데이터를 원래 스케일로 복원하는 함수

    Args:
        original_data (pd.DataFrame): 원본 데이터프레임 (shape: [n_timesteps, n_features])
            - n_timesteps: 타임스텝의 수 (행의 개수)
            - n_features:  특성(feature)의 수 (열의 개수)
        filled_data (pd.DataFrame): 채워진 데이터프레임 (shape: [n_timesteps, n_features])
            - n_timesteps: 타임스텝의 수 (행의 개수)
            - n_features:  특성(feature)의 수 (열의 개수)

    Returns:
        pd.DataFrame: 원래 스케일로 복원된 데이터프레임 (shape: [n_timesteps, n_features])
            - n_timesteps: 타임스텝의 수 (행의 개수)
            - n_features:  특성(feature)의 수 (열의 개수)
    """
    scaled_filled_data = filled_data.copy()  # filled_data의 복사본 생성 (shape: [n_timesteps, n_features])
    columns_with_nan = get_columns_with_nan(original_data)  # 결측치가 있는 열만 선택 (shape: [n_columns_with_nan])
    
    for column in columns_with_nan:
        # 각 열에 대해 원본 데이터의 스케일을 복원
        if original_data[column].dtype.kind in 'biufc':  # 숫자형 데이터 타입 여부 확인
            original_min = original_data[column].min()  # 원본 열의 최소값 (shape: scalar)
            original_max = original_data[column].max()  # 원본 열의 최대값 (shape: scalar)
            filled_min = filled_data[column].min()  # 채워진 열의 최소값 (shape: scalar)
            filled_max = filled_data[column].max()  # 채워진 열의 최대값 (shape: scalar)
            
            # 스케일을 원본 데이터의 스케일로 변환 (shape: [n_timesteps])
            scaled_filled_data[column] = (filled_data[column] - filled_min) / (filled_max - filled_min) * (original_max - original_min) + original_min

    # 최종적으로 원래 스케일로 복원된 데이터프레임 반환 (shape: [n_timesteps, n_features])
    return scaled_filled_data