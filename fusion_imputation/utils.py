import pandas as pd
from typing import List,Tuple
from functools import lru_cache

from functools import lru_cache

@lru_cache(maxsize=None)
def find_columns_with_nan(data_columns: Tuple[str], data_has_nan: Tuple[bool]) -> Tuple[str]:
    """
    NaN 값을 포함한 열을 찾는 함수 (결측치가 있는 열만 선택)

    Args:
        data_columns (Tuple[str]): 데이터프레임의 열 이름들
        data_has_nan (Tuple[bool]): 각 열의 NaN 포함 여부

    Returns:
        Tuple[str]: NaN 값을 포함한 열의 목록
    """
    # zip()으로 data_columns와 data_has_nan을 묶어 순회
    # has_nan이 True인 경우에만 해당 열 이름(col)을 선택
    # 결과를 튜플로 반환하여 불변성 보장
    return tuple(col for col, has_nan in zip(data_columns, data_has_nan) if has_nan)

def get_columns_with_nan(data: pd.DataFrame) -> Tuple[str]:
    """
    데이터프레임에서 NaN 값을 포함하는 열의 이름을 반환하는 함수

    Args:
        data (pd.DataFrame): NaN 값을 검사할 데이터프레임 (shape: [n_timesteps, n_features])
            - n_timesteps: 타임스텝의 수 (행의 개수)
            - n_features: 특성(feature)의 수 (열의 개수)

    Returns:
        Tuple[str]: NaN 값을 포함하는 열 이름들의 튜플 (shape: [n_columns_with_nan])
            - n_columns_with_nan: NaN 값을 포함한 열의 수 (튜플의 길이)

    설명:
        이 함수는 주어진 데이터프레임의 모든 열을 검사하여 NaN 값을 포함하는
        열의 이름을 찾아냅니다. 내부적으로 find_columns_with_nan 함수를 호출하며,
        이 함수는 LRU 캐시를 사용하여 반복적인 호출에 대한 성능을 최적화합니다.
    """
    # 데이터프레임의 열 이름들을 튜플로 변환 (shape: [n_features])
    # data.isna().any()로 각 열의 NaN 포함 여부를 확인하고 튜플로 변환 (shape: [n_features])
    # find_columns_with_nan 함수를 호출하여 NaN을 포함하는 열 이름들을 반환 (shape: [n_columns_with_nan])
    return find_columns_with_nan(tuple(data.columns), tuple(data.isna().any()))