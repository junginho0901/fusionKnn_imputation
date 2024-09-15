import os
import re
from typing import Dict, List
import numpy as np
import pandas as pd

def generate_missing_data(data: pd.DataFrame, missing_rate: float, consecutive_missing_rate: float) -> pd.DataFrame:
    """
    데이터프레임에 결측 값을 생성하는 함수

    Args:
        data (pd.DataFrame): 결측 값을 생성할 데이터프레임 (shape: [n_timesteps, n_features])
            - n_timesteps: 타임스텝의 수 (행의 개수)
            - n_features: 변수 또는 특성(feature)의 수 (열의 개수)
        missing_rate (float): 전체 데이터에서 결측 값의 비율
        consecutive_missing_rate (float): 연속된 결측 값의 길이(비율, 예: 데이터 길이가 10000일 때 0.01이면 100개의 연속 결측값)

    Returns:
        pd.DataFrame: 결측 값이 포함된 데이터프레임 (shape: [n_timesteps, n_features])
            - n_timesteps: 타임스텝의 수 (행의 개수)
            - n_features: 변수 또는 특성(feature)의 수 (열의 개수)
    """
    data_with_missing = data.copy()  # 원본 데이터프레임을 복사 (shape: [n_timesteps, n_features])
    total_values = data.shape[0] * data.shape[1]  # 전체 데이터 포인트의 수 계산 (shape: scalar, n_timesteps * n_features의 결과값)
    num_missing = int(total_values * missing_rate)  # 생성할 결측 값의 총 수 계산 (shape: scalar, 결측 데이터 포인트의 수)
    consecutive_missing_length = max(1, int(data.shape[0] * consecutive_missing_rate))  # 연속 결측 값의 길이 계산 (shape: scalar, 연속된 결측 구간의 길이)

    remaining_missing = num_missing  # 남은 결측 값의 수 (shape: scalar, 남은 결측 데이터 포인트의 수)

    while remaining_missing > 0:
        col = np.random.choice(data_with_missing.columns)  # 랜덤으로 열 선택 (shape: scalar, 선택된 열의 인덱스)
        max_start_index = data.shape[0] - consecutive_missing_length  # 최대 시작 인덱스 계산 (shape: scalar, 가능한 시작 인덱스의 최대값)

        if max_start_index < 0:
            break

        start_index = np.random.randint(0, max_start_index + 1)  # 시작 인덱스 선택 (shape: scalar, 시작할 위치)
        end_index = start_index + consecutive_missing_length  # 끝 인덱스 계산 (shape: scalar, 연속 결측의 끝 인덱스)

        if data_with_missing.iloc[start_index:end_index, data.columns.get_loc(col)].isnull().sum() == 0:
            # 선택된 범위 내에 결측 값이 없을 때만 결측 값 추가
            data_with_missing.iloc[start_index:end_index, data.columns.get_loc(col)] = np.nan  # 해당 열의 연속된 구간에 결측 값 추가 (shape: [consecutive_missing_length], 연속 결측 구간의 길이만큼 결측 값 생성)
            remaining_missing -= consecutive_missing_length  # 남은 결측 값의 수 감소 (shape: scalar)

            if remaining_missing < consecutive_missing_length:
                consecutive_missing_length = remaining_missing  # 남은 결측 값이 연속 길이보다 짧을 경우 조정 (shape: scalar)

    return data_with_missing  # 최종 반환 (shape: [n_timesteps, n_features])

def save_dataframe_to_csv(dataframe: pd.DataFrame, file_path: str) -> None:
    """
    데이터프레임을 CSV 파일로 저장하는 함수

    Args:
        dataframe (pd.DataFrame): 저장할 데이터프레임 (shape: [n_timesteps, n_features])
            - n_timesteps: 타임스텝의 수 (행의 개수)
            - n_features: 변수 또는 특성(feature)의 수 (열의 개수)
        file_path (str): CSV 파일의 경로

    Returns:
        None
    """
    directory = os.path.dirname(file_path)
    if directory:
        os.makedirs(directory, exist_ok=True)  # 디렉토리가 없으면 생성
    dataframe.to_csv(file_path, index=False)  # 데이터프레임을 CSV 파일로 저장 (shape는 [n_timesteps, n_features]로 유지)
    print(f"Dataframe saved to {file_path}")

def get_nan_positions(data_with_missing: pd.DataFrame) -> Dict[int, List[int]]:
    """
    결측 값의 위치를 찾는 함수

    Args:
        data_with_missing (pd.DataFrame): 결측치가 포함된 데이터프레임 (shape: [n_timesteps, n_features])
            - n_timesteps: 타임스텝의 수 (행의 개수)
            - n_features: 변수 또는 특성(feature)의 수 (열의 개수)

    Returns:
        Dict[int, List[int]]: 열 인덱스와 결측 위치 목록을 포함하는 dictionary (shape: {column_index: [missing_indices...]})
            - column_index: 결측 값이 있는 열의 인덱스 (예: 0부터 n_features-1 사이)
            - missing_indices: 해당 열에서 결측 값이 발생한 행 인덱스의 리스트 (예: 0부터 n_timesteps-1 사이)
    """
    nan_positions_dict = {}  # 결측 위치를 저장할 딕셔너리 초기화 (shape: {column_index: [missing_indices...]})
    for i, column in enumerate(data_with_missing.columns):  # 각 열에 대해 반복 (i: 열 인덱스, column: 열 이름)
        # 각 열에 대한 결측 위치를 찾고, 리스트로 저장
        nan_positions = data_with_missing[column][data_with_missing[column].isna()].index.tolist()  # 결측 위치 인덱스 리스트 (shape: [n_missing], 해당 열에서 결측 위치의 수)
        nan_positions_dict[i] = nan_positions  # 각 열 인덱스와 결측 위치 목록을 딕셔너리에 저장 (shape: {column_index: [missing_indices...]})
    return nan_positions_dict  # 최종 반환 (shape: {column_index: [missing_indices...]})
