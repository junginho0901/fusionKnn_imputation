import os
import sys
import time
import pandas as pd
from focalizeKnn_imputation.imputation import focalize_knn_imputation
from common_utils import get_nan_positions
from common_preprocessing import load_and_preprocess_data, load_or_generate_missing_data
from common_evaluation import evaluate_and_save_results, calculate_execution_time
from common_visualization import plot_filled_vs_original, plot_zoomed_filled_vs_original

class FocalizeKNNImputation:
    def __init__(self, data_file_path: str, missing_data_file_path: str, output_dir: str, n_neighbors: int = 5, threshold: float = 0.5, lags: list = [24, 168]):
        """
        Focalize KNN Imputation 초기화 함수

        Args:
            data_file_path (str): 원본 데이터 파일 경로
            missing_data_file_path (str): 결측 데이터 파일 경로
            output_dir (str): 결과 출력 디렉토리 경로
            n_neighbors (int): KNN에서 고려할 이웃의 수 (default: 5)
            threshold (float): 결측값 채우기 시 KNN에서 사용하는 임계값 (default: 0.5)
            lags (list): 시계열 데이터의 시차 리스트 (default: [24, 168])
        """
        self.data_file_path = data_file_path
        self.missing_data_file_path = missing_data_file_path
        self.output_dir = output_dir
        self.n_neighbors = n_neighbors
        self.threshold = threshold 
        self.lags = lags

    def run(self):
        """
        Focalize KNN Imputation을 실행하는 함수

        전체 실행 과정:
        1. 데이터 로드 및 전처리
        2. 결측 데이터 로드 또는 생성
        3. Focalize KNN 보간 수행
        4. 시각화 및 평가 결과 저장
        """
        start_time = time.time()  # 실행 시작 시간 기록

        # 데이터 로드 및 전처리
        df = load_and_preprocess_data(self.data_file_path)  # 원본 데이터 로드 및 전처리 (shape: [n_timesteps, n_features])

        # 결측 데이터 로드 또는 생성
        data_with_missing = load_or_generate_missing_data(df, self.missing_data_file_path)  # 결측 데이터 로드 또는 생성 (shape: [n_timesteps, n_features])

        original_data_with_missing = data_with_missing.copy()  # 원본 데이터의 복사본 저장 (shape: [n_timesteps, n_features])

        # Focalize KNN 보간 수행
        fknn_int_df = focalize_knn_imputation(data_with_missing, self.n_neighbors, self.threshold, self.lags)  # Focalize KNN으로 결측값 채움 (shape: [n_timesteps, n_features])
        fknn_int_df.columns = df.columns  # 채워진 데이터프레임의 열 이름을 원본과 일치시킴 (shape: [n_timesteps, n_features])

        # 그래프 및 평가
        nan_positions_dict = get_nan_positions(original_data_with_missing)  # 결측 위치 정보 추출 (shape: {column_index: [missing_indices...]})
        plot_filled_vs_original(fknn_int_df, df, nan_positions_dict, self.output_dir)  # 채워진 데이터와 원본 데이터 비교 시각화
        plot_zoomed_filled_vs_original(fknn_int_df, df, nan_positions_dict, self.output_dir)  # 확대된 비교 시각화

        # 평가 및 결과 저장
        execution_time_str = calculate_execution_time(start_time)  # 실행 시간 계산
        evaluate_and_save_results(df, fknn_int_df, nan_positions_dict, self.output_dir, os.path.basename(self.data_file_path), len(df.columns), len(df), execution_time_str)  # 평가 결과 저장

        print(execution_time_str)  # 실행 시간 출력
