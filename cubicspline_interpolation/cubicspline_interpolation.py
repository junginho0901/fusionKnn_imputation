import os
import sys
import time
import pandas as pd
from cubicspline_interpolation.interpolation import cubicspline_interpolation
from common_utils import get_nan_positions
from common_preprocessing import load_and_preprocess_data, load_or_generate_missing_data
from common_evaluation import evaluate_and_save_results, calculate_execution_time
from common_visualization import plot_filled_vs_original, plot_zoomed_filled_vs_original

class CubicSplineInterpolation:
    def __init__(self, data_file_path: str, missing_data_file_path: str, output_dir: str):
        """
        Cubic Spline Interpolation을 수행하는 클래스 초기화

        Args:
            data_file_path (str): 원본 데이터 파일 경로
            missing_data_file_path (str): 결측 데이터를 포함한 파일 경로
            output_dir (str): 결과 및 그래프를 저장할 디렉토리 경로
        """
        self.data_file_path = data_file_path
        self.missing_data_file_path = missing_data_file_path
        self.output_dir = output_dir

    def run(self):
        """
        Cubic Spline 보간 및 평가, 결과 저장을 수행하는 메서드
        """
        start_time = time.time()

        # 데이터 로드 및 전처리
        df = load_and_preprocess_data(self.data_file_path)  # 원본 데이터 로드 및 전처리 (shape: [n_timesteps, n_features])
        
        # 결측 데이터 로드 또는 생성
        data_with_missing = load_or_generate_missing_data(df, self.missing_data_file_path)  # 결측 데이터 로드 또는 생성 (shape: [n_timesteps, n_features])

        original_data_with_missing = data_with_missing.copy()  # 결측 데이터를 보관하기 위한 복사본 (shape: [n_timesteps, n_features])

        # Cubic Spline 보간
        spline_int_df = cubicspline_interpolation(data_with_missing)  # Cubic Spline 보간을 통해 결측 값을 채움 (shape: [n_timesteps, n_features])

        # 결측 위치 정보 추출
        nan_positions_dict = get_nan_positions(original_data_with_missing)  # 결측 위치 추출 (shape: {column_index: [missing_indices...]})

        # 보간 결과를 시각화
        plot_filled_vs_original(spline_int_df, df, nan_positions_dict, self.output_dir)  # 전체 데이터와 보간된 데이터를 비교하여 플롯 생성
        plot_zoomed_filled_vs_original(spline_int_df, df, nan_positions_dict, self.output_dir)  # 확대된 플롯 생성

        # 평가 및 결과 저장
        execution_time_str = calculate_execution_time(start_time)  # 실행 시간 계산
        evaluate_and_save_results(df, spline_int_df, nan_positions_dict, self.output_dir, os.path.basename(self.data_file_path), len(df.columns), len(df), execution_time_str)  # 보간 결과 평가 및 저장

        # 실행 시간 출력
        print(execution_time_str)
