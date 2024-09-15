import os
import sys
import time
import pandas as pd
import logging
from fusion_imputation.preprocessing import interpolate_linear, rescale_to_original
from fusion_imputation.imputation import find_closest_features, fill_missing_values_with_similar_segments, generate_segment_data_for_imputation
from fusion_imputation.utils import find_columns_with_nan, get_columns_with_nan
from common_utils import save_dataframe_to_csv, get_nan_positions
from common_preprocessing import load_and_preprocess_data, load_or_generate_missing_data
from common_evaluation import evaluate_and_save_results, calculate_execution_time
from common_visualization import plot_filled_vs_original, plot_zoomed_filled_vs_original

logging.basicConfig(filename='process.log', level=logging.INFO)

class FusionImputation:
    def __init__(self, num_closest_features: int = 5, num_similar_segments: int = 5, dtw_max_warping_segment: int = None, msm_cost: float = 0.1, 
                 segment_step_size: int = 1, segment_random_ratio: float = 1, spearman_exponent: float = 0.5,
                 output_dir: str = "plots", data_file_path: str = None, missing_data_file_path: str = None) -> None:
        """
        FusionImputation 클래스를 초기화하는 메서드

        Args:
            num_closest_features (int): 가장 가까운 특징의 수
            num_similar_segments (int): 선택할 유사 세그먼트의 수
            dtw_max_warping_segment (int): DTW 최대 워핑 세그먼트 크기
            msm_cost (float): MSM 비용
            segment_step_size (int): 세그먼트 슬라이딩 스텝 크기
            segment_random_ratio (float): 세그먼트 랜덤 선택 비율
            spearman_exponent (float): 스피어만 상관계수 지수
            output_dir (str): 결과 저장 디렉토리 경로
            data_file_path (str): 원본 데이터 파일 경로
            missing_data_file_path (str): 결측 데이터 파일 경로
        """
        self.num_closest_features = num_closest_features
        self.num_similar_segments = num_similar_segments
        self.dtw_max_warping_segment = dtw_max_warping_segment
        self.msm_cost = msm_cost
        self.segment_step_size = segment_step_size
        self.segment_random_ratio = segment_random_ratio
        self.spearman_exponent = spearman_exponent
        self.output_dir = output_dir
        self.data_file_path = data_file_path
        self.missing_data_file_path = missing_data_file_path

    def run(self) -> None:
        """
        Fusion Imputation 알고리즘을 실행하는 메서드
        """

        # 전처리 및 결측 데이터 생성-----------------------------------------------------------------------------------
        start_time = time.time()  # 실행 시작 시간 기록

        # 데이터 로드 및 전처리
        df = load_and_preprocess_data(self.data_file_path)  # 원본 데이터 로드 및 전처리 (shape: [n_timesteps, n_features])

        # 결측 데이터 로드 또는 생성
        data_with_missing = load_or_generate_missing_data(df, self.missing_data_file_path)  # 결측 데이터 로드 또는 생성 (shape: [n_timesteps, n_features])

        original_data_with_missing = data_with_missing.copy()  # 원본 데이터 백업 (shape: [n_timesteps, n_features])
        
        # 보간--------------------------------------------------------------------------------------------------------
        # 결측이 존재하는 열과 제일 유사하다고 판단되는 열 딕셔너리 세트로 만들기
        closest_features_dict = find_closest_features(data_with_missing, interpolate_linear(data_with_missing), get_columns_with_nan(data_with_missing), self.num_closest_features, self.dtw_max_warping_segment, self.msm_cost, self.spearman_exponent, self.output_dir)  # 결측 열에 대해 가장 유사한 특징을 찾음 (shape: {column_name: [closest_column_names...]})

        # 유사한 특징들 기반으로 결측값 보완에 사용할 세그먼트 데이터 생성
        segment_data = generate_segment_data_for_imputation(data_with_missing, closest_features_dict, self.num_similar_segments, self.segment_step_size, self.segment_random_ratio, self.output_dir)  # (shape: list of tuples with segment data)

        # 결측값 채우기
        data_with_filled_values = fill_missing_values_with_similar_segments(data_with_missing, segment_data)  # 뽑은 세그먼트를 사용하여 결측값 채움 (shape: [n_timesteps, n_features])
           
        # 결과 및 plot 생성--------------------------------------------------------------------------------------------
        # 원본 스케일로 재변환 및 저장
        scaled_filled_data = rescale_to_original(df, data_with_filled_values)  # 원본 스케일로 데이터 재변환 (shape: [n_timesteps, n_features])
        save_dataframe_to_csv(scaled_filled_data, os.path.join(self.output_dir, "scaled_filled_data.csv"))  # CSV 파일로 저장

        # 그래프 및 평가
        nan_positions_dict = get_nan_positions(original_data_with_missing)  # 결측 위치 찾기 (shape: {column_index: [missing_indices...]})
        plot_filled_vs_original(scaled_filled_data, df, nan_positions_dict, self.output_dir)  # 원본 데이터와 채워진 데이터 비교 시각화
        plot_zoomed_filled_vs_original(scaled_filled_data, df, nan_positions_dict, self.output_dir)  # 확대된 비교 시각화

        # 평가 및 결과 저장
        execution_time_str = calculate_execution_time(start_time)  # 실행 시간 계산
        evaluate_and_save_results(df, data_with_filled_values, nan_positions_dict, self.output_dir, os.path.basename(self.data_file_path), len(df.columns), len(df), execution_time_str)  # 평가 및 결과 저장

        print(execution_time_str)  # 실행 시간 출력