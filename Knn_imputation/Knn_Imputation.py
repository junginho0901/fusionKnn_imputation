import os
import sys
import time
import pandas as pd
from Knn_imputation.imputation import knn_imputation
from common_utils import get_nan_positions
from common_preprocessing import load_and_preprocess_data, load_or_generate_missing_data
from common_evaluation import evaluate_and_save_results, calculate_execution_time
from common_visualization import plot_filled_vs_original, plot_zoomed_filled_vs_original

class KNNImputation:
    def __init__(self, data_file_path: str, missing_data_file_path: str, output_dir: str, n_neighbors: int = 5):
        """
        KNN 보간을 위한 초기화 함수

        Args:
            data_file_path (str): 원본 데이터 파일 경로
            missing_data_file_path (str): 결측 데이터 파일 경로
            output_dir (str): 출력 디렉토리 경로
            n_neighbors (int): KNN의 이웃 수
        """
        self.data_file_path = data_file_path
        self.missing_data_file_path = missing_data_file_path
        self.output_dir = output_dir
        self.n_neighbors = n_neighbors

    def run(self) -> None:
        """
        KNN 보간 프로세스를 실행하는 함수
        """
        start_time = time.time()  # 시작 시간 기록

        # 데이터 로드 및 전처리
        df = load_and_preprocess_data(self.data_file_path)  # (shape: [n_timesteps, n_features])
        
        # 결측 데이터 로드 또는 생성
        data_with_missing = load_or_generate_missing_data(df, self.missing_data_file_path)  # (shape: [n_timesteps, n_features])
        
        original_data_with_missing = data_with_missing.copy()  # 원본 데이터 백업 (shape: [n_timesteps, n_features])

        # KNN 보간 수행
        knn_int_df = knn_imputation(data_with_missing, self.n_neighbors)  # (shape: [n_timesteps, n_features])
        knn_int_df.columns = df.columns  # 열 이름을 원본 데이터프레임과 일치시킴 (shape: [n_timesteps, n_features])

        # 결측 위치 찾기 및 시각화
        nan_positions_dict = get_nan_positions(original_data_with_missing)  # (shape: {column_index: [missing_indices...]})
        plot_filled_vs_original(knn_int_df, df, nan_positions_dict, self.output_dir)  # 원본과 보간된 데이터 비교 시각화
        plot_zoomed_filled_vs_original(knn_int_df, df, nan_positions_dict, self.output_dir)  # 확대된 비교 시각화

        # 평가 및 결과 저장
        execution_time_str = calculate_execution_time(start_time)  # 실행 시간 계산
        evaluate_and_save_results(df, knn_int_df, nan_positions_dict, self.output_dir, os.path.basename(self.data_file_path), len(df.columns), len(df), execution_time_str)  # 평가 및 결과 저장

        print(execution_time_str)  # 실행 시간 출력


# if __name__ == '__main__':
#     data_path = '/home/ih.jeong/new_folder/_dataset/data0723/test.csv'
#     missing_data_path = '/home/ih.jeong/new_folder/_dataset/data0723/test_missing.csv'
#     output_dir = 'knn_test_0.5_0.03'

#     imputer = KNNImputation(
#         data_file_path=data_path,
#         missing_data_file_path=missing_data_path,
#         output_dir=output_dir,
#         n_neighbors=5
#     )
#     imputer.run()
