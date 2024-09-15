import os
from typing import Dict, List
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_filled_vs_original(filled_data: pd.DataFrame, original_data: pd.DataFrame, nan_positions_dict: Dict[int, List[int]], output_dir: str) -> None:
    """
    채워진 데이터와 원본 데이터를 비교하여 시각화하는 함수

    Args:
        filled_data (pd.DataFrame): 방법론을 통하여 채워진 데이터프레임 (shape: [n_timesteps, n_features])
            - n_timesteps: 타임스텝의 수 (행의 개수)
            - n_features: 변수 또는 특성(feature)의 수 (열의 개수)
        original_data (pd.DataFrame): 원본 데이터프레임 (shape: [n_timesteps, n_features])
            - n_timesteps: 타임스텝의 수 (행의 개수)
            - n_features: 변수 또는 특성(feature)의 수 (열의 개수)
        nan_positions_dict (Dict[int, List[int]]): 결측값이 존재하던 위치 dictionary (shape: {column_index: [missing_indices...]})
            - column_index: 결측값이 포함된 열의 인덱스 (0부터 시작)
            - missing_indices: 결측값이 발생한 행 인덱스의 리스트 (해당 열에서의 인덱스)
        output_dir (str): 출력 디렉토리 경로

    Returns:
        None
    """
    # 출력 디렉토리 설정 및 생성
    output_dir = os.path.join(output_dir, "imputation_plots")
    os.makedirs(output_dir, exist_ok=True)
    
    # 각 열(컬럼)별로 원본 데이터와 채워진 데이터를 비교하여 시각화
    for i, filled_column in enumerate(filled_data.columns):
        original_column = original_data.columns[i]
        
        # 새로운 플롯 생성 (크기: 15x5)
        plt.figure(figsize=(15, 5))
        
        # 원본 데이터를 파란색으로 플롯 (shape: [n_timesteps])
        # - n_timesteps: 시간 축에 따라 기록된 데이터 포인트의 수 (해당 열의 모든 값)
        plt.plot(original_data[original_column], label='Original Data', color='blue', alpha=0.7)
        
        # 결측 위치를 가져옴 (해당 열에 대해)
        nan_positions = nan_positions_dict.get(i, [])
        if nan_positions:
            nan_positions_list = nan_positions
            # 결측 위치를 그룹화하여 연속된 결측 구간을 찾음
            groups = [[nan_positions_list[0]]]
            for idx in nan_positions_list[1:]:
                if idx == groups[-1][-1] + 1:
                    groups[-1].append(idx)
                else:
                    groups.append([idx])
            # 각 그룹에 대해 결측 구간을 시각적으로 강조
            for group in groups:
                plt.axvspan(group[0] - 0.5, group[-1] + 0.5, color='red', alpha=0.2)
            
            # 채워진 데이터 중 결측 구간에 해당하는 부분만 플롯 (shape: [n_timesteps])
            # - n_timesteps: 시간 축에 따라 기록된 데이터 포인트의 수 (해당 열의 모든 값)
            filled_only = filled_data[filled_column].copy()
            filled_only[~filled_only.index.isin(nan_positions_list)] = np.nan
            plt.plot(filled_only, label='Filled Data', color='red', linestyle='dashed', alpha=0.7)

        # 그래프의 제목, 축 레이블, 범례 설정
        plt.title(f'Data for {filled_column}')
        plt.xlabel('Index')
        plt.ylabel('Value')
        plt.legend()
        # 저장
        plt.savefig(os.path.join(output_dir, f'{filled_column}_comparison.png'))
        plt.close()

def plot_zoomed_filled_vs_original(filled_data: pd.DataFrame, original_data: pd.DataFrame, nan_positions_dict: Dict[int, List[int]], output_dir: str) -> None:
    """
    채워진 데이터와 원본 데이터를 비교하여 확대 시각화하는 함수

    Args:
        filled_data (pd.DataFrame): 방법론을 통하여 채워진 데이터프레임 (shape: [n_timesteps, n_features])
            - n_timesteps: 타임스텝의 수 (행의 개수)
            - n_features: 변수 또는 특성(feature)의 수 (열의 개수)
        original_data (pd.DataFrame): 원본 데이터프레임 (shape: [n_timesteps, n_features])
            - n_timesteps: 타임스텝의 수 (행의 개수)
            - n_features: 변수 또는 특성(feature)의 수 (열의 개수)
        nan_positions_dict (Dict[int, List[int]]): 결측값이 존재하던 위치 dictionary (shape: {column_index: [missing_indices...]})
            - column_index: 결측값이 포함된 열의 인덱스 (0부터 시작)
            - missing_indices: 결측값이 발생한 행 인덱스의 리스트 (해당 열에서의 인덱스)
        output_dir (str): 출력 디렉토리 경로

    Returns:
        None
    """
    # 출력 디렉토리 설정 및 생성
    output_dir = os.path.join(output_dir, "zoomed_imputation_plots")
    os.makedirs(output_dir, exist_ok=True)
    
    # 각 열(컬럼)별로 원본 데이터와 채워진 데이터를 비교하여 확대된 시각화
    for i, filled_column in enumerate(filled_data.columns):
        original_column = original_data.columns[i]
        
        # 결측 위치를 가져옴 (해당 열에 대해)
        nan_positions = nan_positions_dict.get(i, [])
        if not nan_positions:
            continue
        
        # 결측 위치를 그룹화하여 연속된 결측 구간을 찾음
        nan_groups = []
        current_group = [nan_positions[0]]
        for idx in nan_positions[1:]:
            if idx == current_group[-1] + 1:
                current_group.append(idx)
            else:
                nan_groups.append(current_group)
                current_group = [idx]
        nan_groups.append(current_group)
        
        # 각 결측 그룹에 대해 확대된 시각화 진행
        for group in nan_groups:
            start_idx = group[0]
            end_idx = group[-1]
            gap_length = end_idx - start_idx + 1
            # 결측 구간 앞뒤로 동일한 길이만큼의 데이터 포함하여 확대
            start = max(0, start_idx - gap_length)
            end = min(len(original_data), end_idx + gap_length + 1)
            
            plt.figure(figsize=(15, 5))
            
            # 원본 데이터를 파란색으로 플롯 (shape: [end-start])
            # - end-start: 확대된 구간의 데이터 포인트 수 (해당 열의 시간 구간에 따라 선택된 값들)
            plt.plot(original_data[original_column].iloc[start:end], label='Original Data', color='blue', alpha=0.7)

            # 채워진 데이터 중 결측 구간에 해당하는 부분만 플롯 (shape: [end-start])
            # - end-start: 확대된 구간의 데이터 포인트 수 (해당 열의 시간 구간에 따라 선택된 값들)
            filled_only = filled_data[filled_column].iloc[start:end].copy()
            not_nan_indices = ~filled_only.index.isin(group)
            filled_only[not_nan_indices] = np.nan
            plt.plot(filled_only, label='Filled Data', color='red', linestyle='dashed', alpha=0.7)
            
            # 결측 구간을 시각적으로 강조
            plt.axvspan(group[0] - 0.5, group[-1] + 0.5, color='red', alpha=0.2)
            
            # 그래프의 제목, 축 레이블, 범례 설정
            plt.title(f'Zoomed Data for {filled_column} around position {start_idx} to {end_idx}')
            plt.xlabel('Index')
            plt.ylabel('Value')
            plt.legend()
            
            # 파일 이름을 안전하게 변경하여 플롯을 저장
            plt.savefig(os.path.join(output_dir, f'{filled_column}_zoomed_{start_idx}_{end_idx}.png'))
            plt.close()
