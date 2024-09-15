import os
import sys
import time
import pandas as pd
from knndtw_imputation.imputation import KNN_DTW
from common_utils import get_nan_positions
from common_preprocessing import load_and_preprocess_data, load_or_generate_missing_data
from common_evaluation import evaluate_and_save_results, calculate_execution_time
from common_visualization import plot_filled_vs_original, plot_zoomed_filled_vs_original

class KNNDTWImputation:
    def __init__(self, data_file_path: str, missing_data_file_path: str, output_dir: str, n_neighbors: int = 8):
        """
        KNN-DTW 보간을 수행하는 클래스

        Args:
            data_file_path (str): 원본 데이터 파일 경로
            missing_data_file_path (str): 결측 데이터 파일 경로
            output_dir (str): 출력 디렉토리
            n_neighbors (int): KNN의 이웃 수 (default: 8)
        """
        self.data_file_path = data_file_path
        self.missing_data_file_path = missing_data_file_path
        self.output_dir = output_dir
        self.n_neighbors = n_neighbors

    def run(self) -> None:
        """
        KNN-DTW 보간 실행 메서드
        """
        start_time = time.time()  # 시작 시간 기록

        # 데이터 로드 및 전처리
        df = load_and_preprocess_data(self.data_file_path)  # (shape: [n_timesteps, n_features])

        # 결측 데이터 로드 또는 생성
        data_with_missing = load_or_generate_missing_data(df, self.missing_data_file_path)  # (shape: [n_timesteps, n_features])

        original_data_with_missing = data_with_missing.copy()  # 원본 데이터 백업 (shape: [n_timesteps, n_features])

        # KNN-DTW 보간
        interpolated_data = data_with_missing.copy()  # 보간할 데이터의 복사본 생성 (shape: [n_timesteps, n_features])
        
        for column in interpolated_data.columns:
            nan_indices = interpolated_data[column][interpolated_data[column].isna()].index  # 결측 위치 인덱스 추출 (shape: [n_missing])
            known_indices = interpolated_data[column][~interpolated_data[column].isna()].index  # 값이 있는 인덱스 추출 (shape: [n_known])
            
            if len(known_indices) > 0 and len(nan_indices) > 0:
                # KNN-DTW 모델 초기화 및 학습
                knn_dtw = KNN_DTW(n_neighbors=self.n_neighbors)  # KNN-DTW 객체 생성
                knn_dtw.fit(known_indices.values.reshape(-1, 1), interpolated_data.loc[known_indices, column].values)  # 모델 학습
                # 결측 위치에 대해 예측 수행 및 값 채우기
                interpolated_data.loc[nan_indices, column] = knn_dtw.predict(nan_indices.values.reshape(-1, 1))  # 보간 수행 (shape: [n_missing])

        # 그래프 및 평가
        nan_positions_dict = get_nan_positions(original_data_with_missing)  # 결측 위치 찾기 (shape: {column_index: [missing_indices...]})
        plot_filled_vs_original(interpolated_data, df, nan_positions_dict, self.output_dir)  # 원본 데이터와 채워진 데이터 비교 시각화
        plot_zoomed_filled_vs_original(interpolated_data, df, nan_positions_dict, self.output_dir)  # 확대된 비교 시각화

        # 평가 및 결과 저장
        execution_time_str = calculate_execution_time(start_time)  # 실행 시간 계산
        evaluate_and_save_results(df, interpolated_data, nan_positions_dict, self.output_dir, os.path.basename(self.data_file_path), len(df.columns), len(df), execution_time_str)  # 평가 및 결과 저장

        print(execution_time_str)  # 실행 시간 출력
