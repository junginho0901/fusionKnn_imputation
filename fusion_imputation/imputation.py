from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from sktime.distances import dtw_distance, msm_distance
from fusion_imputation.visualization import plot_baseline_segments, plot_closest_features
from fusion_imputation.preprocessing import interpolate_linear
from numpy.lib.stride_tricks import as_strided

def find_closest_features(data_with_missing: pd.DataFrame, data_interpolated: pd.DataFrame, columns_with_nan: List[str], num_closest_features: int, dtw_max_warping_segment: int, msm_cost: float, spearman_exponent: float, output_dir: str) -> Dict[str, List[str]]:
    """
    결측 값을 포함한 열에 대해 가장 가까운 특징을 찾는 함수

    Args:
        data_with_missing (pd.DataFrame): 결측 값이 포함된 데이터프레임 (shape: [n_timesteps, n_features])
            - n_timesteps: 타임스텝의 수 (행의 개수)
            - n_features:  특성(feature)의 수 (열의 개수)
        data_interpolated (pd.DataFrame): (선형)보간된 데이터프레임 (shape: [n_timesteps, n_features])
            - n_timesteps: 타임스텝의 수 (행의 개수)
            - n_features:  특성(feature)의 수 (열의 개수)
        columns_with_nan (List[str]): 결측 값을 포함한 열의 목록 (shape: [n_columns_with_nan])
            - n_columns_with_nan: 결측값이 포함된 열의 수
        num_closest_features (int): 가장 가까운 feature의 수
        dtw_max_warping_segment (int, optional): DTW 최대 워핑 segment 크기
        msm_cost (float): MSM 비용
        spearman_exponent (float): 스피어만 상관계수 지수
        output_dir (str): plot 출력 디렉토리

    Returns:
        Dict[str, List[str]]: 열 이름과 가장 가까운 특징의 목록을 포함하는 딕셔너리 (shape: {column_name: [closest_column_names...]})
            - column_name: 결측값이 포함된 열의 이름
            - closest_column_names: 가장 가까운 특징을 가진 열의 이름들로 구성된 리스트
    """
    closest_features_dict = {}
    
    # 모든 열 쌍에 대해 한 번에 스피어만 상관계수 계산 (shape: [n_features, n_features])
    correlation_matrix = data_interpolated.corr(method='spearman')
    
    for missing_column_name in columns_with_nan:
        # 각 결측 값을 포함한 열에 대해 다른 열들과의 거리를 계산
        distance_map = {}
        missing_data = data_interpolated[missing_column_name].values  # (shape: [n_timesteps])
        
        for col in data_interpolated.columns:
            if col != missing_column_name:
                # 두 열의 데이터 (shape: [n_timesteps])에 대한 스피어만 상관계수 계산
                # n_timesteps: 시간 축을 따라 기록된 데이터 포인트의 수
                spearman_corr = correlation_matrix.loc[missing_column_name, col]
                
                if spearman_corr < 0.5:
                    # 상관계수가 낮으면 거리를 무한대로 설정
                    distance_map[col] = np.inf
                else:
                    other_data = data_interpolated[col].values  # (shape: [n_timesteps])
                    # MSM 거리 계산 및 로그 변환 (shape: scalar)
                    msm_distance_val = msm_distance(missing_data, other_data, c=msm_cost)
                    log_msm_distance = np.log(msm_distance_val + 1)
                    # DTW 거리 계산 및 로그 변환 (shape: scalar)
                    dtw_distance_val = dtw_distance(missing_data, other_data, segment=dtw_max_warping_segment)
                    log_dtw_distance = np.log(dtw_distance_val + 1)
                    # 결합 거리 계산 (스피어만 상관계수로 가중치 부여, shape: scalar)
                    combined_distance = (log_msm_distance + log_dtw_distance) * (1 / abs(spearman_corr) ** spearman_exponent)
                    distance_map[col] = combined_distance

        # 계산된 거리들을 기준으로 정렬 (shape: [(column_name, distance)] * n_features)
        sorted_features = sorted(distance_map.items(), key=lambda x: x[1])
        # 무한대가 아닌 가장 가까운 features를 선택 (shape: [(column_name, distance)] * num_closest_features)
        closest_features_with_distance = [(feature, dist) for feature, dist in sorted_features if dist != np.inf][:num_closest_features]

        # 본인 자신을 feature에 추가(본인 자신의 과거, 미래 데이터도 결측 보간의 데이터가 되어야 하기 때문)
        closest_features_with_distance.append((missing_column_name, None))

        # 결측치 위치 찾기
        nan_indices = data_with_missing[missing_column_name][data_with_missing[missing_column_name].isna()].index
        # 해당 열과 가장 가까운 features을 그린 plot 생성
        plot_closest_features(data_interpolated, missing_column_name, closest_features_with_distance, nan_indices, output_dir)
        print(f"Plotted closest features for {missing_column_name}")
        # 최종적으로 가장 가까운 features를 dict에 저장 (shape: {column_name: [closest_column_names...]})
        closest_features_dict[missing_column_name] = [feature for feature, _ in closest_features_with_distance]

    return closest_features_dict

def generate_segment_data_for_imputation(data: pd.DataFrame, closest_features_dict: Dict[str, List[str]], num_similar_segments: int, segment_step_size: int, segment_random_ratio: float, output_dir: str) -> List[Tuple[str, int, np.ndarray, np.ndarray, np.ndarray, List[str], np.ndarray, np.ndarray]]:
    """
    # 가장 가까운 특징을 사용하여 세그먼트 리스트 생성

    Args:
        data (pd.DataFrame): 데이터프레임(결측치가 존재하던 csv 파일을 DataFrame 형태로 만든 것) (shape: [n_timesteps, n_features])
            - n_timesteps: 타임스텝의 수 (행의 개수)
            - n_features:  특성(feature)의 수 (열의 개수)
        closest_features_dict (Dict[str, List[str]]): 가장 similar한 특징들 딕셔너리
            - str: 결측치를 포함하는 열 이름
            - List[str]: 가장 유사한 열 이름들의 리스트
        num_similar_segments (int): 선택할 유사 세그먼트의 수
        segment_step_size (int): 세그먼트 슬라이딩 스텝 크기
        segment_random_ratio (float): 세그먼트 랜덤 선택 비율
        output_dir (str): 결과 저장 디렉토리 경로

    Returns:
        List[Tuple[str, int, np.ndarray, np.ndarray, np.ndarray, List[str], np.ndarray, np.ndarray]]:
            이 함수는 결측치를 포함하는 각 열에 대해 처리된 정보를 포함하는 튜플의 리스트를 반환
            각 튜플은 다음 요소들을 포함:
            - str: 결측치를 포함하는 열 이름
            - int: 결측치를 포함하는 세그먼트의 시작 인덱스
            - np.ndarray: 결측치를 포함하는 세그먼트 (shape: [segment_size])
            - np.ndarray: 결측 위치 목록 (shape: [n_nan_positions])
            - np.ndarray: 선택된 세그먼트 세트 (shape: [n_selected_segments, segment_size])
            - List[str]: 선택된 세그먼트 세트와 관련된 열 이름들 (shape: [n_selected_segments])
            - np.ndarray: top 인덱스 (shape: [num_similar_segments])
            - np.ndarray: top 세그먼트와의 거리 (shape: [num_similar_segments])
            - n_selected_segments: 선택된 세그먼트 세트의 수
            - segment_size: 각 세그먼트의 길이
            - n_nan_positions: 각 세그먼트 내의 결측 위치 수
            - num_similar_segments: top-k 세그먼트의 수
    """
    segment_data = []
    
    for missing_column, similar_columns in closest_features_dict.items():
        # 각 열에 대해 결측값이 있는 세그먼트 크기를 계산 (최대 연속 결측값 길이의 3배)
        nan_mask = data[missing_column].isna()
        nan_groups = np.split(np.where(nan_mask)[0], np.where(np.diff(np.where(nan_mask)[0]) != 1)[0] + 1)
        segment_size = min(data.shape[0], max(map(len, nan_groups)) * 3)
        
        # 결측값이 포함된 세그먼트를 슬라이딩 윈도우로 생성 (shape: [n_nan_segments, segment_size])
        missing_segments, missing_indices, missing_positions_list = sliding_windows_with_nan(data[missing_column], segment_size)
        
        similar_segments_list = []  # 여러 특징들의 세그먼트들을 한번에 저장할 리스트
        similar_column_names = []  # 해당 열 이름을 저장할 리스트
        
        for similar_column in similar_columns:
            if similar_column in data.columns:
                # 유사한 열에 대해 슬라이딩 윈도우로 세그먼트 생성 (shape: [n_segments_per_column, segment_size])
                similar_segments = sliding_windows(data[similar_column].values, segment_size, segment_step_size)
                similar_segments_list.append(similar_segments)  # 생성된 세그먼트를 리스트에 추가
                similar_column_names.extend([similar_column] * len(similar_segments))

        # 모든 유사 열의 세그먼트를 결합 (shape: [total_n_segments, segment_size])
        all_similar_segments = np.vstack(similar_segments_list)
        
        num_segments_to_select = int(segment_random_ratio * len(all_similar_segments))  # 선택할 세그먼트 수 계산
        selected_indices = np.random.choice(len(all_similar_segments), num_segments_to_select, replace=False)  # 랜덤으로 선택된 인덱스
        selected_segments_set = all_similar_segments[selected_indices]  # 선택된 세그먼트 세트 (shape: [n_selected_segments, segment_size])
        selected_column_names = [similar_column_names[i] for i in selected_indices]  # 선택된 열 이름 (shape: [n_selected_segments])
        
        # 선택된 세그먼트 세트와 결측 세그먼트 사이의 거리 행렬 계산 (shape: [n_nan_segments, n_selected_segments])
        distance_matrix = calculate_distance_matrix(missing_segments, selected_segments_set, missing_positions_list, missing_column, output_dir)

        for row_index, row in enumerate(distance_matrix):
            # 거리를 기준으로 정렬 (shape: [n_selected_segments])
            sorted_indices = np.argsort(row)
            sorted_values = row[sorted_indices]
            top_indices = sorted_indices[:num_similar_segments]  # top-k 인덱스 선택 (shape: [num_similar_segments])
            top_values = sorted_values[:num_similar_segments]  # top-k 거리 값 선택 (shape: [num_similar_segments])
            # 최종적으로 각 결측 세그먼트에 대해 처리된 정보를 segment_data에 추가
            segment_data.append((missing_column, missing_indices[row_index], missing_segments[row_index], missing_positions_list[row_index], selected_segments_set, selected_column_names, top_indices, top_values))

    

    # 최종적으로 결측치를 처리하는데 필요한 세그먼트 데이터 리스트 반환 (shape: [(column_name, segment_start_idx, segment, nan_positions, selected_segments, selected_column_names, top_indices, top_values)] * n_nan_segments)
    return segment_data


def sliding_windows(series: pd.Series, segment_size: int, segment_step_size: int) -> np.ndarray:
    """
    주어진 시리즈에 대해 슬라이딩 윈도우를 하여 segment 세트를 생성하는 함수

    Args:
        series (pd.Series): time 시리즈 데이터 (shape: [n_timesteps])
            - n_timesteps: 시간의 흐름에 따라 기록된 데이터 포인트 수
        segment_size (int): 슬라이딩 윈도우할 때의 윈도우 크기
        segment_step_size (int): 슬라이딩 윈도우의 step 크기

    Returns:
        np.ndarray: 슬라이딩 윈도우를 한 결과 segment들이 저장된 배열 (shape: [n_segments, segment_size])
            - n_segments: 슬라이딩 윈도우를 통해 생성된 segment의 수
            - segment_size: 각 segment의 길이
    """
    if len(series) < segment_size:
        return np.array([])
    
    n = ((len(series) - segment_size) // segment_step_size) + 1
    return as_strided(series, shape=(n, segment_size), 
                      strides=(series.strides[0] * segment_step_size, series.strides[0]))


def sliding_windows_with_nan(series: pd.Series, segment_size: int) -> Tuple[np.ndarray, List[int], List[np.ndarray]]:
    """
    결측 값을 포함한 슬라이딩 윈도우를 해서 segment set를 생성하는 함수
    이 함수는 주어진 시계열 데이터에서 결측값(NaN)이 포함된 구간을 중심으로 슬라이딩 윈도우를 적용하여
    결측값을 포함한 세그먼트를 추출

    Args:
        series (pd.Series): 시리즈 데이터 (shape: [n_timesteps])
            - n_timesteps: 시간의 흐름에 따라 기록된 데이터 포인트 수
        segment_size (int): 슬라이딩 윈도우 크기

    Returns:
        Tuple[np.ndarray, List[int], List[np.ndarray]]: 
        - 슬라이딩 윈도우를 통한 세그먼트 배열 (shape: [n_segments, segment_size])
            - n_segments: 슬라이딩 윈도우를 통해 생성된 segment의 수
            - segment_size: 각 segment의 길이
        - 인덱스 목록 (shape: [n_segments])
            - n_segments: 각 segment의 시작 인덱스의 수
        - 결측 위치 목록 (shape: [n_segments, n_nan_positions])
            - n_segments: 각 segment 내에서 NaN 위치의 수
            - n_nan_positions: 각 segment 내에서 NaN 위치의 수
    """
    half_segment = segment_size // 2  # 슬라이딩 윈도우의 절반 크기
    segments = []  # 결과 세그먼트들을 저장할 리스트
    indices = []  # 각 세그먼트의 시작 인덱스를 저장할 리스트
    nan_positions_list = []  # 각 세그먼트에서 결측값이 있는 위치를 저장할 리스트

    is_nan = series.isna()  # 결측값이 있는 위치를 True, 없는 위치를 False로 변환 (shape: [n_timesteps])
    nan_indices = np.where(is_nan)[0]  # NaN 값이 있는 인덱스를 추출 (shape: [n_nan_indices])

    if len(nan_indices) == 0:
        # NaN이 없을 경우, 빈 리스트 반환
        return np.array([]), [], []

    # NaN 값이 연속되지 않은 그룹들을 찾기 위한 로직
    gaps = np.diff(nan_indices) > 1  # 연속되지 않은 NaN 값들 사이의 갭을 찾음 (shape: [n_nan_indices-1])
    gap_indices = np.where(gaps)[0]  # 갭 위치의 인덱스를 추출 (shape: [n_gaps])
    groups = np.split(nan_indices, gap_indices + 1)  # 갭을 기준으로 NaN 그룹 분할 (shape: List of arrays of varying lengths)

    for group in groups:
        start_idx = group[0]  # 그룹의 첫 번째 NaN 인덱스
        end_idx = group[-1]  # 그룹의 마지막 NaN 인덱스
        center_idx = (start_idx + end_idx) // 2  # 그룹의 중심 인덱스 계산

        segment_start = max(0, center_idx - half_segment)  # 중심 인덱스를 기준으로 세그먼트의 시작 위치 계산
        segment_end = min(len(series), segment_start + segment_size)  # 세그먼트의 끝 위치 계산

        if segment_end - segment_start < segment_size:
            # 세그먼트의 크기가 부족할 경우, 시작 위치를 조정
            segment_start = segment_end - segment_size

        if segment_start >= 0 and segment_end <= len(series):
            # 계산된 세그먼트가 유효할 경우, 세그먼트 추가
            segment = series[segment_start:segment_end]
            segments.append(segment.values)  # 세그먼트를 리스트에 추가 (shape: [segment_size])
            indices.append(segment_start)  # 세그먼트의 시작 인덱스를 저장 (shape: scalar)
            nan_positions_list.append(np.where(np.isnan(segment))[0])  # 세그먼트 내의 NaN 위치를 저장 (shape: [n_nan_positions])

    # 최종적으로 각 세그먼트에 대해 n_segments, segment_size, 그리고 NaN 위치 정보를 반환 (shape: [n_segments, segment_size], [n_segments], [n_segments, n_nan_positions])
    return np.array(segments), indices, nan_positions_list


def calculate_baseline_distance(segment1: np.ndarray, segment2: np.ndarray, nan_positions: np.ndarray) -> float:
    """
    baseline segment와 결측치 세그먼트 사이의 유클리드 거리 계산하는 함수

    Args:
        segment1 (np.ndarray): 결측 segment 배열 (shape: [segment_size])
            - segment_size: 세그먼트의 길이
        segment2 (np.ndarray): baseline segment 배열 (shape: [segment_size])
            - segment_size: 세그먼트의 길이
        nan_positions (np.ndarray): 결측 위치가 저장된 배열 (shape: [n_nan_positions])
            - n_nan_positions: 세그먼트 내의 결측 위치 수

    Returns:
        float: 두 segment 사이의 거리 값 (shape: scalar)
            - scalar: 단일 값 (거리)
    """
    # 유효한 인덱스를 선택 (결측치가 아닌 위치)
    valid_indices = [i for i in range(len(segment1)) if i not in nan_positions and not np.isnan(segment1[i]) and not np.isnan(segment2[i])]
    
    if len(valid_indices) == 0:
        # 유효한 인덱스가 없으면 무한대 거리 반환
        return np.inf
    
    # 유효한 인덱스에 해당하는 값을 추출 (shape: [n_valid_indices])
    valid_segment1 = segment1[valid_indices]
    valid_segment2 = segment2[valid_indices]
    
    # 두 세그먼트 간의 유클리드 거리 계산 (shape: scalar)
    return np.linalg.norm(valid_segment1 - valid_segment2)

def calculate_partial_euclidean_distance(missing_segment: np.ndarray, comparison_segment: np.ndarray, missing_positions: np.ndarray) -> float:
    """
    결측 위치를 고려한 부분 유클리드 거리를 계산하는 함수

    Args:
        missing_segment (np.ndarray): 결측치를 포함한 세그먼트 배열 (shape: [segment_size])
            - segment_size: 세그먼트의 길이
        comparison_segment (np.ndarray): 비교 대상 세그먼트 배열 (shape: [segment_size])
            - segment_size: 세그먼트의 길이
        missing_positions (np.ndarray): 결측 위치가 저장된 배열 (shape: [n_nan_positions])
            - n_nan_positions: 세그먼트 내의 결측 위치 수

    Returns:
        float: 두 세그먼트 사이의 유클리드 거리 값 (shape: scalar)
            - scalar: 단일 값 (거리)
    """
    if np.isscalar(comparison_segment):
        # comparison_segment가 스칼라 값인 경우 무한대 반환 (배열 간 유클리드 계산이 불가)
        return np.inf
    
    if np.isnan(comparison_segment).any():
        # comparison_segment에 결측값이 있으면 무한대 반환 (결측값이 있는 세그먼트는 유효하지 않다고 간주)
        return np.inf
    
    missing_mask = np.isnan(missing_segment)
    # missing_segment에서 결측값이 있는 위치를 찾기 위한 마스크 생성 (shape: [segment_size])
    
    valid_positions = [i for i in range(len(missing_segment)) if i not in missing_positions and not missing_mask[i]]
    # missing_segment에서 결측값이 아닌 위치의 인덱스들을 리스트로 저장
    # valid_positions는 missing_positions에도 포함되지 않고, missing_mask에도 포함되지 않는 인덱스들의 리스트
    
    if len(valid_positions) == 0:
        # 유효한 인덱스가 없으면 무한대 거리 반환
        return np.inf
    
    # 유효한 인덱스에 해당하는 값을 추출 (shape: [n_valid_positions])
    valid_missing_segment = missing_segment[valid_positions]
    # missing_segment에서 유효한 인덱스에 해당하는 값들을 추출
    
    valid_comparison_segment = comparison_segment[valid_positions]
    # comparison_segment에서 유효한 인덱스에 해당하는 값들을 추출
    
    # 두 세그먼트 간의 유클리드 거리 계산 (shape: scalar)
    return np.linalg.norm(valid_missing_segment - valid_comparison_segment)
    # 추출된 유효한 값들 간의 유클리드 거리를 계산하여 반환


def calculate_distance_matrix(missing_segments: np.ndarray, complete_segments: np.ndarray, missing_positions_list: List[np.ndarray], key: str, output_dir: str) -> np.ndarray:
    """
    결측 창과 정상 창(sliding window로 생성된 segment들) 간의 거리를 계산하는 함수

    Args:
        missing_segments (np.ndarray): 결측 segment 배열 (shape: [n_nan_segments, segment_size])
            - n_nan_segments: 결측 세그먼트의 수
            - segment_size: 각 세그먼트의 길이
        complete_segments (np.ndarray): 정상 segment 배열 (shape: [n_normal_segments, segment_size])
            - n_normal_segments: 정상 세그먼트의 수
            - segment_size: 각 세그먼트의 길이
        missing_positions_list (List[np.ndarray]): 결측 위치 목록 (shape: [n_nan_segments, n_nan_positions])
            - n_nan_segments: 결측 세그먼트의 수
            - n_nan_positions: 각 세그먼트 내의 결측 위치 수
        key (str): 열 이름
        output_dir (str): plot 출력 디렉토리

    Returns:
        np.ndarray: 거리가 저장된 행렬 (shape: [n_nan_segments, n_normal_segments])
            - n_nan_segments: 결측 세그먼트의 수
            - n_normal_segments: 정상 세그먼트의 수
    """
    distance_matrix = np.zeros((len(missing_segments), len(complete_segments)))  # 거리 행렬 초기화 (shape: [n_nan_segments, n_normal_segments])
    baseline_segments = create_linear_baseline_segments(missing_segments, missing_positions_list)  # linear 보간 baseline 세그먼트 생성 (shape: [n_nan_segments, segment_size])
    
    plot_baseline_segments(missing_segments, baseline_segments, missing_positions_list, key, output_dir)  # baseline 세그먼트 시각화

    for i, (missing_segment, baseline_segment, missing_positions) in enumerate(zip(missing_segments, baseline_segments, missing_positions_list)):
        baseline_segment_distance = calculate_baseline_distance(missing_segment, baseline_segment, missing_positions)  # baseline 세그먼트와 결측치가 있는 세그먼트 사이의 거리 계산 : 베이스라인 값 (shape: scalar)
        
        for j, complete_segment in enumerate(complete_segments):
            distance = calculate_partial_euclidean_distance(missing_segment, complete_segment, missing_positions)  # 정상 세그먼트와의 거리 계산 (shape: scalar)
            distance_matrix[i, j] = distance if distance < baseline_segment_distance else np.inf  # baseline 거리보다 작으면 거리 값을 저장, 그렇지 않으면 무한대
    
    # 최종 거리 행렬 반환 (shape: [n_nan_segments, n_normal_segments])
    return distance_matrix


# def create_linear_baseline_segments(nan_segments: np.ndarray, nan_positions_list: List[np.ndarray]) -> np.ndarray:
#     """
#     linear 보간 baseline 세그먼트를 생성하는 함수
 
#     Args:
#         nan_segments (np.ndarray): 결측 세그먼트 배열 (shape: [n_nan_segments, segment_size])
#             - n_nan_segments: 결측 세그먼트의 수
#             - segment_size: 각 세그먼트의 길이
#         nan_positions_list (List[np.ndarray]): 결측 위치 목록 (shape: [n_nan_segments, n_nan_positions])
#             - n_nan_segments: 결측 세그먼트의 수
#             - n_nan_positions: 각 세그먼트 내의 결측 위치 수

#     Returns:
#         np.ndarray: linear 보간 segment 배열 (shape: [n_nan_segments, segment_size])
#             - n_nan_segments: 결측 세그먼트의 수
#             - segment_size: 각 세그먼트의 길이
#     """
#     linear_baseline_segments = []
#     for segment, nan_positions in zip(nan_segments, nan_positions_list):
#         linear_segment = np.full_like(segment, np.nan)  # NaN으로 채워진 새로운 segment 생성 (shape: [segment_size])
        
#         non_nan_segments = []  # NaN이 아닌 구간을 저장할 리스트
#         start = 0
#         for i in range(len(segment) + 1):
#             if i == len(segment) or i in nan_positions:
#                 if start < i:
#                     non_nan_segments.append((start, i))  # NaN이 아닌 구간의 시작과 끝을 리스트에 저장
#                 start = i + 1
        
#         for start, end in non_nan_segments:
#             if end - start > 1:
#                 start_value = segment[start]
#                 end_value = segment[end - 1]
#                 slope = (end_value - start_value) / (end - start - 1)  # 구간의 선형 보간 계산
#                 for i in range(start, end):
#                     linear_segment[i] = start_value + slope * (i - start)  # 보간된 값으로 채움
#             elif end - start == 1:
#                 linear_segment[start] = segment[start]  # 단일 값인 경우 그대로 복사
        
#         linear_baseline_segments.append(linear_segment)  # 리스트에 추가 (shape: [segment_size])
    
#     # 최종적으로 각 세그먼트에 대해 NaN을 선형 보간한 결과 반환 (shape: [n_nan_segments, segment_size])
#     return np.array(linear_baseline_segments)

def create_linear_baseline_segments(nan_segments: np.ndarray, nan_positions_list: List[np.ndarray]) -> np.ndarray:
    """
    연속된 결측값 구간을 기준으로 세그먼트를 나누고, 각 구간별로 선형 변환을 적용하는 함수
 
    Args:
        nan_segments (np.ndarray): 결측 세그먼트 배열 (shape: [n_nan_segments, segment_size])
        nan_positions_list (List[np.ndarray]): 결측 위치 목록 (shape: [n_nan_segments, n_nan_positions])

    Returns:
        np.ndarray: 수정된 baseline segment 배열 (shape: [n_nan_segments, segment_size])
    """
    linear_baseline_segments = []
    for segment, nan_positions in zip(nan_segments, nan_positions_list):
        linear_segment = segment.copy()
        
        # 연속된 NaN 구간 찾기
        nan_ranges = []
        start = None
        for i in range(len(segment)):
            if i in nan_positions:
                if start is None:
                    start = i
            elif start is not None:
                nan_ranges.append((start, i))
                start = None
        if start is not None:
            nan_ranges.append((start, len(segment)))
        
        # NaN이 아닌 구간에 대해 선형 변환 적용
        valid_ranges = [(0, nan_ranges[0][0])] if nan_ranges else [(0, len(segment))]
        for i in range(len(nan_ranges) - 1):
            valid_ranges.append((nan_ranges[i][1], nan_ranges[i+1][0]))
        if nan_ranges:
            valid_ranges.append((nan_ranges[-1][1], len(segment)))
        
        for start, end in valid_ranges:
            if start != end:
                start_value = segment[start]
                end_value = segment[end - 1]
                if not np.isnan(start_value) and not np.isnan(end_value):
                    slope = (end_value - start_value) / (end - start - 1)
                    for i in range(start, end):
                        linear_segment[i] = start_value + slope * (i - start)
        
        linear_baseline_segments.append(linear_segment)
    
    return np.array(linear_baseline_segments)

def combine_segment_sets(segment_sets: List[np.ndarray], column_names: List[str]) -> Tuple[np.ndarray, List[str]]:
    """
    segment 세트를 결합하는 함수(dictionary value들의 각각의 segment set를 하나의 set(일렬)로 결합)

    Args:
        segment_sets (List[np.ndarray]): segment 세트 들 목록 (shape: [n_columns, n_segments_per_column, segment_size])
            - n_columns: segment를 생성한 열의 수
            - n_segments_per_column: 각 열에서 생성된 세그먼트의 수
            - segment_size: 각 세그먼트의 길이
        column_names (List[str]): 열 이름 목록 (shape: [n_columns])
            - n_columns: segment를 생성한 열의 수

    Returns:
        Tuple[np.ndarray, List[str]]: 결합된 segment 배열과 열 이름 목록 (shape: [total_n_segments, segment_size], [total_n_segments])
            - total_n_segments: 모든 열에서 생성된 세그먼트의 총 수
            - segment_size: 각 세그먼트의 길이
    """
    combined_segment_sets = np.vstack(segment_sets)  # 모든 세그먼트를 하나의 배열로 결합 (shape: [total_n_segments, segment_size])
    combined_column_names = [col for col, segments in zip(column_names, segment_sets) for _ in range(segments.shape[0])]  # 각 세그먼트에 해당하는 열 이름을 저장 (shape: [total_n_segments])
    return combined_segment_sets, combined_column_names

def fill_missing_values_with_similar_segments(data: pd.DataFrame, segment_data: List[Tuple[str, int, np.ndarray, np.ndarray, np.ndarray, List[str], np.ndarray, np.ndarray]]) -> pd.DataFrame:
    """
    유사한 세그먼트를 사용하여 결측 값을 채우는 함수

    Args:
        data (pd.DataFrame): 결측값이 포함된 데이터프레임 (shape: [n_timesteps, n_features])
            - n_timesteps: 타임스텝의 수 (행의 개수)
            - n_features: 특성(feature)의 수 (열의 개수)
        segment_data (List[Tuple[str, int, np.ndarray, np.ndarray, np.ndarray, List[str], np.ndarray, np.ndarray]]): 세그먼트들에 대한 데이터
            - Tuple의 각 요소는 다음과 같음:
                - str: 결측치를 포함하는 열 이름 (column_name)
                - int: 결측치를 포함하는 세그먼트의 시작 인덱스 (segment_start_index)
                - np.ndarray: 결측치를 포함하는 세그먼트 (shape: [segment_size])
                - np.ndarray: 결측 위치 목록 (shape: [n_nan_positions])
                - np.ndarray: 선택된 세그먼트 세트 (shape: [n_selected_segments, segment_size])
                - List[str]: 선택된 세그먼트 세트와 관련된 열 이름들 (shape: [n_selected_segments])
                - np.ndarray: top 인덱스 (shape: [num_similar_segments])
                - np.ndarray: top 세그먼트와의 거리 (shape: [num_similar_segments])

    Returns:
        pd.DataFrame: 결측 값이 채워진 데이터프레임 (shape: [n_timesteps, n_features])
            - n_timesteps: 타임스텝의 수 (행의 개수)
            - n_features: 특성(feature)의 수 (열의 개수)
    """
    for column_name, segment_start_index, _, relative_nan_positions, selected_segments_set, _, top_indices, top_values in segment_data:
        # column_name: 결측치가 포함된 열의 이름
        # segment_start_index: 결측치가 포함된 세그먼트의 시작 인덱스
        # relative_nan_positions: 세그먼트 내에서 결측치가 있는 상대적 위치들
        # selected_segments_set: 결측치가 있는 세그먼트와 유사한 세그먼트들
        # top_indices: 유사한 세그먼트들 중 상위 k개의 인덱스
        # top_values: 유사한 세그먼트들과의 거리

        imputation_results = []  # 나중에 채워질 값들을 저장할 리스트
        
        for relative_nan_position in relative_nan_positions:
            # 각 결측 위치에 대해 처리
            absolute_nan_position = segment_start_index + relative_nan_position  # 데이터프레임 내의 실제 인덱스 계산
            valid_similar_indices = [idx for idx, val in enumerate(top_values) if val != np.inf]  # 무한대가 아닌 유효한 거리 값들만 선택
            
            if len(valid_similar_indices) == 0:
                # 유사한 세그먼트가 없는 경우 (즉, 모두 무한대인 경우) 건너뜀
                continue
            else:
                valid_similar_distances = np.array(top_values)[valid_similar_indices]  # 유사한 거리 값들을 추출 (shape: [n_valid_indices])
                similar_segments = selected_segments_set[top_indices[valid_similar_indices]]  # 유사한 세그먼트들을 선택 (shape: [n_valid_indices, segment_size])
                
                squared_distances = valid_similar_distances ** 2 + 1e-10  # 거리 값 제곱 후 작은 값을 더하여 계산 (shape: [n_valid_indices])
                # 1e-10을 더하는 이유는 distance에서 0이 나올 시 밑의 과정에서 0으로 나누는 것을 방지하기 위함
                inverse_distance_weights = 1 / squared_distances  # 거리에 반비례하는 가중치 계산 (shape: [n_valid_indices])
                normalized_weights = inverse_distance_weights / np.sum(inverse_distance_weights)  # 가중치 합이 1이 되도록 정규화 (shape: [n_valid_indices])
                
                similar_segment_values = [segment[relative_nan_position] for segment in similar_segments]  # 유사한 세그먼트에서 결측 위치에 대한 값을 추출 (shape: [n_valid_indices])
                weighted_average = np.dot(normalized_weights, similar_segment_values)  # 가중치를 사용하여 결측 값을 계산 (shape: scalar)
            
                imputation_results.append((absolute_nan_position, column_name, weighted_average))  # 계산된 값을 리스트에 추가
        
        for absolute_nan_position, column_name, weighted_average in imputation_results:
            # 계산된 값을 실제 데이터프레임에 채워 넣음
            data.at[absolute_nan_position, column_name] = weighted_average  # data.at은 데이터프레임의 특정 위치에 값을 할당하는 메서드
    
    data = interpolate_linear(data) # 베이스 라인(선형)을 넘지 못하여 세그먼트로 보간하지 못한 부분을 선형 보간 적용 (shape: [n_timesteps, n_features])
    # 최종적으로 결측값이 채워진 데이터프레임을 반환 (shape: [n_timesteps, n_features])
    return data

