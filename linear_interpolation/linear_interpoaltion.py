import os
import sys
import time
import pandas as pd
from linear_interpolation.imputation import linear_interpolation
from common_utils import get_nan_positions
from common_preprocessing import load_and_preprocess_data, load_or_generate_missing_data
from common_evaluation import evaluate_and_save_results, calculate_execution_time
from common_visualization import plot_filled_vs_original, plot_zoomed_filled_vs_original

class LinearInterpolation:
    def __init__(self, data_file_path: str, missing_data_file_path: str, output_dir: str):
        self.data_file_path = data_file_path  # 데이터 파일 경로
        self.missing_data_file_path = missing_data_file_path  # 결측 데이터 파일 경로
        self.output_dir = output_dir  # 출력 디렉토리 경로

    def run(self) -> None:
        start_time = time.time()  # 시작 시간 기록

        # 데이터 로드 및 전처리
        df = load_and_preprocess_data(self.data_file_path)  # (shape: [n_timesteps, n_features])

        # 결측 데이터 로드 또는 생성
        data_with_missing = load_or_generate_missing_data(df, self.missing_data_file_path)  # (shape: [n_timesteps, n_features])

        original_data_with_missing = data_with_missing.copy()  # 원본 데이터 백업 (shape: [n_timesteps, n_features])

        # 선형 보간
        li_int_df = linear_interpolation(data_with_missing)  # (shape: [n_timesteps, n_features])

        # 그래프 및 평가
        nan_positions_dict = get_nan_positions(original_data_with_missing)  # 결측 위치 정보 (shape: {column_index: [missing_indices...]})
        plot_filled_vs_original(li_int_df, df, nan_positions_dict, self.output_dir)  # 보간 결과와 원본 데이터 비교 시각화
        plot_zoomed_filled_vs_original(li_int_df, df, nan_positions_dict, self.output_dir)  # 확대된 비교 시각화

        # 평가 및 결과 저장
        execution_time_str = calculate_execution_time(start_time)  # 실행 시간 계산
        evaluate_and_save_results(df, li_int_df, nan_positions_dict, self.output_dir, os.path.basename(self.data_file_path), len(df.columns), len(df), execution_time_str)  # 평가 및 결과 저장

        print(execution_time_str)  # 실행 시간 출력
