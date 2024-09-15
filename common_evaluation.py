from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
import time
import os

def evaluate_and_save_results(scaled_df: pd.DataFrame, filled_df: pd.DataFrame, nan_positions_dict: Dict[int, List[int]], output_dir: str, data_file_name: str, num_columns: int, num_rows: int, execution_time_str: str) -> None:
    """
    보간된 데이터 결과를 평가하고 저장하는 함수

    Args:
        scaled_df (pd.DataFrame): 스케일링된 원본 데이터프레임 (shape: [n_timesteps, n_features])
            - n_timesteps: 타임스텝의 수 (행의 개수)
            - n_features: feature의 개수 (열의 개수)
        filled_df (pd.DataFrame): 보간된 데이터프레임 (shape: [n_timesteps, n_features])
            - n_timesteps: 타임스텝의 수 (행의 개수)
            - n_features: feature의 개수 (열의 개수)
        nan_positions_dict (Dict[int, List[int]]): 결측 위치 정보 (shape: {column_index: [missing_indices...]})
            - column_index: 데이터프레임의 열 인덱스 (정수)
            - missing_indices: 각 열에서 결측치가 발생한 행 인덱스들의 리스트 (정수 리스트)
        output_dir (str): 결과를 저장할 디렉토리 경로
        data_file_name (str): 원본 데이터 파일 이름
        num_columns (int): 데이터프레임의 컬럼 수 (데이터프레임의 열 개수)
        num_rows (int): 데이터프레임의 행 수 (데이터프레임의 행 개수)
        execution_time_str (str): 실행 시간 문자열 (형식화된 실행 시간)

    Returns:
        None
    """
    # 보간된 데이터에 대한 MSE 평가
    mse_values, total_mse = evaluate_imputation(scaled_df, filled_df, nan_positions_dict)
    print("MSE values:", mse_values)
    print("Total MSE:", total_mse)

    # 평가 결과를 저장할 파일 경로 설정
    mse_file_path = os.path.join(output_dir, "evaluation_results.txt")
    # 평가 결과를 텍스트 파일로 저장
    save_evaluation_results(mse_file_path, data_file_name, num_columns, num_rows, mse_values, total_mse, execution_time_str)

def calculate_execution_time(start_time: float) -> str:
    """
    코드 실행 시간을 계산하는 함수

    Args:
        start_time (float): 코드 실행 시작 시간 (epoch time, float)

    Returns:
        str: 실행 시간을 시/분/초 형식으로 반환 (형식화된 실행 시간 문자열)
    """
    end_time = time.time()  # 현재 시간을 기록하여 종료 시간 설정 (float)
    execution_time = end_time - start_time  # 실행 시간 계산 (float)

    # 실행 시간을 시간, 분, 초 단위로 변환
    hours, rem = divmod(execution_time, 3600)
    minutes, seconds = divmod(rem, 60)
    # 형식화된 실행 시간 문자열을 반환
    return f"Total execution time: {int(hours):02d}:{int(minutes):02d}:{seconds:05.2f}"

def evaluate_imputation(original_data: pd.DataFrame, filled_data: pd.DataFrame, nan_positions_dict: Dict[int, List[int]]) -> Tuple[Dict[str, float], float]:
    """
    결측 값 채우기 평가 지표를 계산하는 함수

    Args:
        original_data (pd.DataFrame): 원본 데이터프레임 (shape: [n_timesteps, n_features])
            - n_timesteps: 타임스텝의 수 (행의 개수)
            - n_features: feature의 개수 (열의 개수)
        filled_data (pd.DataFrame): 채워진 데이터프레임 (shape: [n_timesteps, n_features])
            - n_timesteps: 타임스텝의 수 (행의 개수)
            - n_features: feature의 개수 (열의 개수)
        nan_positions_dict (Dict[int, List[int]]): 결측 위치 딕셔너리 (shape: {column_index: [missing_indices...]})
            - column_index: 데이터프레임의 열 인덱스 (정수)
            - missing_indices: 각 열에서 결측치가 발생한 행 인덱스들의 리스트 (정수 리스트)

    Returns:
        Tuple[Dict[str, float], float]: 열별 MSE 값 딕셔너리와 총 MSE 값
            - mse_values (Dict[str, float]): 각 열의 이름을 키로 하고, 해당 열의 MSE 값을 값으로 가지는 딕셔너리 (shape: {column_name: mse_value})
                - column_name: 열 이름 (문자열)
                - mse_value: 해당 열의 MSE 값 (실수)
            - total_mse (float): 모든 열의 MSE 값을 합산한 총 MSE 값 (실수)
    """
    mse_values = {}  # 각 열에 대한 MSE 값을 저장할 딕셔너리 초기화

    # 데이터프레임의 각 열에 대해 MSE를 계산
    for i in range(len(original_data.columns)):
        original_column = original_data.columns[i]  # 현재 처리 중인 원본 데이터의 열 이름
        filled_column = filled_data.columns[i]  # 현재 처리 중인 채워진 데이터의 열 이름
        
        if i in nan_positions_dict and nan_positions_dict[i]:
            # 결측 위치에 해당하는 원본 값과 보간된 값을 가져옴 (shape: [n_missing_values])
            nan_indices = nan_positions_dict[i]
            original_values = original_data[original_column].iloc[nan_indices].values  # 원본 데이터에서 결측치가 있던 위치의 값 (shape: [n_missing_values])
            filled_values = filled_data[filled_column].iloc[nan_indices].values  # 채워진 데이터에서 해당 위치의 값 (shape: [n_missing_values])
            
            # MSE 계산 (shape: scalar)
            mse = mean_squared_error(original_values, filled_values)
            
            # MSE 값을 딕셔너리에 저장 (shape: {column_name: mse_value})
            mse_values[original_column] = mse

    # 총 MSE 계산 (shape: scalar)
    total_mse = sum(mse_values.values())

    # 열별 MSE 값과 총 MSE 값을 반환 (shape: {column_name: mse_value}, scalar)
    return mse_values, total_mse

def save_evaluation_results(file_path: str, original_file_name: str, num_features: int, num_data_points: int, mse_values: Dict[str, float], total_mse: float, execution_time_str: str) -> None:
    """
    평가 결과를 텍스트 파일로 저장하는 함수

    Args:
        file_path (str): 저장할 파일 경로 (문자열)
        original_file_name (str): 원본 파일 이름 (문자열)
        num_features (int): 데이터프레임의 특징 수 (열 수, 정수)
        num_data_points (int): 데이터 포인트 수 (행 수, 정수)
        mse_values (Dict[str, float]): 열별 MSE 값 (shape: {column_name: mse_value})
            - column_name: 열 이름 (문자열)
            - mse_value: 해당 열의 MSE 값 (실수)
        total_mse (float): 총 MSE (실수, 모든 열의 MSE 합계)
        execution_time_str (str): 실행 시간 문자열 (형식화된 실행 시간)

    Returns:
        None
    """
    # 텍스트 파일을 열고 쓰기 모드로 설정
    with open(file_path, 'w', encoding='utf-8') as mse_file:
        # 원본 파일 이름, 특징 수, 데이터 포인트 수를 파일에 기록
        mse_file.write(f"Original file name: {original_file_name}\n")
        mse_file.write(f"Number of features: {num_features}\n")
        mse_file.write(f"Number of data points: {num_data_points}\n\n")
        mse_file.write("MSE values:\n")
        # 각 열에 대한 MSE 값을 파일에 기록
        for key, value in mse_values.items():
            mse_file.write(f"{key}: {value}\n")
        # 총 MSE와 실행 시간을 파일에 기록
        mse_file.write(f"\nTotal MSE: {total_mse}\n")
        mse_file.write(f"\n{execution_time_str}\n")
    # 파일 저장 완료 후 메시지 출력
    print(f"Evaluation results saved to {file_path}")
