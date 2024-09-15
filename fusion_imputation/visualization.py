import os
import sys
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def plot_closest_features(data: pd.DataFrame, target_col: str, closest_features_with_distance: List[Tuple[str, float]], nan_indices: pd.Index, output_dir: str) -> None:
    """
    뽑아낸 가까운 특징들을 시각화하는 함수

    Args:
        data (pd.DataFrame): 전체 데이터프레임 (shape: [n_timesteps, n_features])
            - n_timesteps: 시간의 흐름에 따라 기록된 타임스텝의 수 (행의 개수)
            - n_features: 각 타임스텝에서 기록된 변수 또는 특성(feature)의 수 (열의 개수)
        target_col (str): 해당 열 이름(딕셔너리의 key 이름)
        closest_features_with_distance (List[Tuple[str, float]]): 가장 가까운 특징과 거리를 포함하는 목록 (shape: [n_closest_features, 2])
            - n_closest_features: 타겟 열에 대해 가장 유사한 특징(feature)의 수
            - 각 튜플의 첫 번째 요소는 열 이름 (str), 두 번째 요소는 해당 열과의 거리 (float)
        nan_indices (pd.Index): 결측 값이 있는 인덱스 (shape: [n_nan_indices])
            - n_nan_indices: 타겟 열에서 결측치(NaN)가 발생한 타임스텝의 수 (인덱스의 개수)
        output_dir (str): 출력 디렉토리 경로

    Returns:
        None
    """
    # 출력 디렉토리 경로를 설정, "plot_feature_dictionary"라는 하위 디렉토리 생성
    output_dir = os.path.join(output_dir, "plot_feature_dictionary")
    os.makedirs(output_dir, exist_ok=True)  # 디렉토리가 존재하지 않으면 생성
    
    # 플롯의 크기를 설정 (가로 15인치, 세로 5인치)
    plt.figure(figsize=(15, 5))
    # 타겟 열(target_col)의 데이터를 플롯에 그리며, 빨간색으로 표시
    plt.plot(data[target_col], label=f'{target_col} (with NaNs)', color='red')  # (shape: [n_timesteps])

    # 가까운 특징 목록을 순회하며 각 특징을 플롯에 추가
    for feature, distance in closest_features_with_distance:
        if distance is not None:
            # 거리가 있는 경우: 특징의 데이터와 함께 거리 정보를 레이블로 추가
            plt.plot(data[feature], label=f'{feature} (distance={distance:.2f})', linestyle='dashed')  # (shape: [n_timesteps])
        else:
            # 거리 정보가 없는 경우: 단순히 특징의 데이터만 플롯에 추가
            plt.plot(data[feature], label=f'{feature}', linestyle='dashed')  # (shape: [n_timesteps])

    # 결측 위치(nan_indices)가 있는 경우 시각적으로 강조
    if len(nan_indices) > 0:
        nan_groups = []  # 연속된 결측 위치를 그룹화할 리스트 (shape: List of List[int])
        current_group = [nan_indices[0]]  # 첫 번째 결측 위치를 그룹에 추가 (shape: [1])

        # 결측 위치를 순회하며 연속된 인덱스를 그룹화
        for idx in nan_indices[1:]:
            if idx == current_group[-1] + 1:
                # 현재 그룹의 마지막 인덱스와 연속된 경우 그룹에 추가
                current_group.append(idx)  # (shape: [n_group_items])
            else:
                # 연속되지 않는 경우 현재 그룹을 리스트에 추가하고 새 그룹 시작
                nan_groups.append(current_group)  # (shape: [n_groups, n_group_items])
                current_group = [idx]

        # 마지막 그룹을 리스트에 추가
        nan_groups.append(current_group)  # (shape: [n_groups, n_group_items])

        # 각 그룹을 시각적으로 강조 (붉은색 반투명 영역으로 표시)
        for group in nan_groups:
            plt.axvspan(group[0], group[-1], color='red', alpha=0.2)

    # 플롯의 제목, 축 레이블, 범례를 설정
    plt.title(f'Closest features for {target_col}')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend()

    # 플롯을 파일로 저장 
    plt.savefig(f"{output_dir}/{target_col}_closest_features.png")
    plt.close()  # 플롯을 닫아 메모리 해제

def plot_baseline_segments(nan_segments: np.ndarray, linear_baseline_segments: np.ndarray, nan_positions_list: List[np.ndarray], key: str, output_dir: str) -> None:
    """
    원본 segment와 선형 baseline segment를 같이 시각화하는 함수

    Args:
        nan_segments (np.ndarray): 결측 값이 포함된 segment 배열 (shape: [n_segments, segment_size])
            - n_segments: 결측치를 포함한 세그먼트의 수
            - segment_size: 각 세그먼트의 길이 (타임스텝의 수)
        linear_baseline_segments (np.ndarray): 선형 baseline segment 배열 (shape: [n_segments, segment_size])
            - n_segments: 결측치를 포함한 세그먼트의 수
            - segment_size: 각 세그먼트의 길이 (타임스텝의 수)
        nan_positions_list (List[np.ndarray]): 결측 위치 목록 (shape: [n_segments, n_nan_positions])
            - n_segments: 결측치를 포함한 세그먼트의 수
            - n_nan_positions: 각 세그먼트에서 결측값이 발생한 위치의 수
        key (str): 열 이름
        output_dir (str): 출력 디렉토리 경로

    Returns:
        None
    """
    # 출력 디렉토리 경로를 설정, "baseline_segments"라는 하위 디렉토리 생성
    output_dir = os.path.join(output_dir, "baseline_segments")
    os.makedirs(output_dir, exist_ok=True)  # 디렉토리가 존재하지 않으면 생성
    
    # 각 세그먼트를 순회하며 시각화
    for i, (nan_segment, linear_baseline, nan_positions) in enumerate(zip(nan_segments, linear_baseline_segments, nan_positions_list)):
        plt.figure(figsize=(15, 5))  # 플롯의 크기 설정
        # 원본 세그먼트를 플롯에 추가 (결측값 포함, 빨간색으로 표시)
        plt.plot(nan_segment, label='Original segment with NaNs', color='red', alpha=0.7)  # (shape: [segment_size])
        # 선형 보간된 세그먼트를 플롯에 추가 (파란색, 점선으로 표시)
        plt.plot(linear_baseline, label='Linear baseline', color='blue', linestyle='--')  # (shape: [segment_size])
        
        # 결측 위치를 시각적으로 강조
        if len(nan_positions) > 0:
            groups = [[nan_positions[0]]]  # 연속된 결측 위치를 그룹화할 리스트 (shape: List of List[int])
            for pos in nan_positions[1:]:
                if pos == groups[-1][-1] + 1:
                    # 현재 그룹의 마지막 인덱스와 연속된 경우 그룹에 추가
                    groups[-1].append(pos)  # (shape: [n_group_items])
                else:
                    # 연속되지 않는 경우 새 그룹을 추가
                    groups.append([pos])  # (shape: [n_groups, n_group_items])
            # 각 그룹을 시각적으로 강조 (붉은색 반투명 영역으로 표시)
            for group in groups:
                plt.axvspan(group[0] - 0.5, group[-1] + 0.5, color='red', alpha=0.3)
        
        # 플롯의 제목, 축 레이블, 범례를 설정
        plt.title(f'Baseline segment for {key} - segment {i+1}')
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.legend()

        # 플롯을 파일로 저장
        plt.savefig(f"{output_dir}/{key}_baseline_segment_{i+1}.png")
        plt.close()  # 플롯을 닫아 메모리 해제

def plot_segments(segment_data: List[Tuple[str, int, np.ndarray, np.ndarray, np.ndarray, List[str], np.ndarray, np.ndarray]], num_similar_segments: int, output_dir: str) -> None:
    """
    top-k로 뽑힌 segment들을 시각화하는 함수

    Args:
        segment_data (List[Tuple[str, int, np.ndarray, np.ndarray, np.ndarray, List[str], np.ndarray, np.ndarray]]): segment 데이터 (shape: [n_segments, segment_size])
            - 각 튜플의 첫 번째 요소: 열 이름 (str)
            - 두 번째 요소: 세그먼트 시작 인덱스 (int)
            - 세 번째 요소: 결측치를 포함하는 세그먼트 (np.ndarray) (shape: [segment_size])
            - 네 번째 요소: 결측치 위치 목록 (np.ndarray) (shape: [n_nan_positions])
            - 다섯 번째 요소: 선택된 세그먼트 세트 (np.ndarray) (shape: [n_selected_segments, segment_size])
            - 여섯 번째 요소: 선택된 세그먼트들이 속한 열 이름 리스트 (List[str]) (shape: [n_selected_segments])
            - 일곱 번째 요소: 선택된 세그먼트의 인덱스들 (np.ndarray) (shape: [num_similar_segments])
            - 여덟 번째 요소: 선택된 세그먼트들과의 거리 (np.ndarray) (shape: [num_similar_segments])
        num_similar_segments (int): 시각화할 상위 유사 세그먼트의 개수
        output_dir (str): 출력 디렉토리 경로

    Returns:
        None
    """
    # 출력 디렉토리 경로를 설정, "plots_segment_set"라는 하위 디렉토리 생성
    output_dir = os.path.join(output_dir, "plots_segment_set")
    os.makedirs(output_dir, exist_ok=True)  # 디렉토리가 존재하지 않으면 생성
    
    # 각 세그먼트를 순회하며 시각화
    for idx, (key, nan_index, nan_segment, nan_positions, selected_segments_set, selected_column_names, top_indices, top_values) in enumerate(segment_data):
        plt.figure(figsize=(15, 5))  # 플롯의 크기 설정
        
        # 플롯의 다양한 선 스타일과 두께 설정
        line_styles = ['dashed', 'dotted', 'dashdot', (0, (3, 5, 1, 5)), (0, (5, 10))]
        line_widths = [1, 1.5, 2, 2.5, 3]

        # top-k 유사 세그먼트를 시각화
        for i in range(num_similar_segments):
            style = line_styles[i % len(line_styles)]  # 선 스타일 설정
            width = line_widths[i % len(line_widths)]  # 선 두께 설정
            plt.plot(selected_segments_set[top_indices[i]], label=f'Top {i+1} matching segment (index={top_indices[i]}, column={selected_column_names[top_indices[i]]}, dist={top_values[i]:.3f})', linestyle=style, linewidth=width)  # (shape: [segment_size])
        
        # 결측 값이 포함된 세그먼트를 시각화 (빨간색으로 표시)
        plt.plot(nan_segment, label=f'{key} with NaNs', color='red', linewidth=2)  # (shape: [segment_size])
        
        # 결측 위치를 시각적으로 강조
        if len(nan_positions) > 0:
            groups = [[nan_positions[0]]]  # 연속된 결측 위치를 그룹화할 리스트 (shape: List of List[int])
            for pos in nan_positions[1:]:
                if pos == groups[-1][-1] + 1:
                    # 현재 그룹의 마지막 인덱스와 연속된 경우 그룹에 추가
                    groups[-1].append(pos)  # (shape: [n_group_items])
                else:
                    # 연속되지 않는 경우 새 그룹을 추가
                    groups.append([pos])  # (shape: [n_groups, n_group_items])
            # 각 그룹을 시각적으로 강조 (붉은색 반투명 영역으로 표시)
            for group in groups:
                plt.axvspan(group[0] - 0.5, group[-1] + 0.5, color='red', alpha=0.3)
        
        # 플롯의 제목, 축 레이블, 범례를 설정
        plt.title(f'segments for {key}')
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.legend()

        # 플롯을 파일로 저장 (파일명은 sanitize_filename 함수로 처리된 key 이름 사용)
        plt.savefig(f"{output_dir}/{key}_segment_{idx}_plot.png")
        plt.close()  # 플롯을 닫아 메모리 해제

