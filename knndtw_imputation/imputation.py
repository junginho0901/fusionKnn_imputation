from typing import Dict, List
import numpy as np
import pandas as pd
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
from sklearn.neighbors import KNeighborsRegressor

def dtw_distance(ts_a: np.ndarray, ts_b: np.ndarray) -> float:
    """
    DTW(Dynamic Time Warping) 거리를 계산하는 함수

    Args:
        ts_a (np.ndarray): 시계열 데이터 A (shape: [n_timesteps_a])
        ts_b (np.ndarray): 시계열 데이터 B (shape: [n_timesteps_b])

    Returns:
        float: 두 시계열 데이터 간의 DTW 거리 (shape: scalar)
    """
    ts_a, ts_b = np.array(ts_a), np.array(ts_b)
    distance, path = fastdtw(ts_a.reshape(-1, 1), ts_b.reshape(-1, 1), dist=euclidean)  # DTW 거리 계산
    return distance  # 최종적으로 계산된 거리 반환

class KNN_DTW(KNeighborsRegressor):
    def __init__(self, n_neighbors=1):
        """
        KNN-DTW 모델 초기화

        Args:
            n_neighbors (int): KNN의 이웃 수

        Returns:
            None
        """
        # KNeighborsRegressor의 초기화를 통해 KNN 모델 생성
        # metric 매개변수에 dtw_distance 함수를 사용하여, DTW 거리를 기반으로 이웃을 찾음
        super().__init__(n_neighbors=n_neighbors, metric=dtw_distance)
