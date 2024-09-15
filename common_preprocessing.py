import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import os
from common_utils import generate_missing_data, save_dataframe_to_csv

def load_and_preprocess_data(file_path: str) -> pd.DataFrame:
    """
    데이터를 로드하고 전처리하는 함수. 
    데이터 파일에서 특정 컬럼을 제거하고 컬럼명을 정리하며, 데이터를 0-1 범위로 스케일링함.

    Args:
        file_path (str): 데이터 파일 경로

    Returns:
        pd.DataFrame: 전처리된 데이터프레임 (shape: [n_timesteps, n_features])
            - n_timesteps: 타임스텝의 수 (행의 개수)
            - n_features: feature의 개수 (열의 개수)
    """
    try:
        # 데이터를 'latin1' 인코딩으로 읽어들이려고 시도
        df = pd.read_csv(file_path, encoding='latin1')  # (shape: [n_timesteps, n_features_before_processing])
        print(f"File read successfully from {file_path} with 'latin1' encoding.")
    except UnicodeDecodeError as e:
        # 'latin1' 인코딩 실패 시 'ISO-8859-1' 인코딩으로 다시 시도
        print(f"Error: {e}")
        df = pd.read_csv(file_path, encoding='ISO-8859-1')  # (shape: [n_timesteps, n_features_before_processing])
        print(f"File read successfully from {file_path} with 'ISO-8859-1' encoding.")
    
    # 첫 번째 열이 타임스탬프일 경우 제거
    df = remove_first_column_if_timestamp(df)  # (shape: [n_timesteps, n_features_after_timestamp_removal])
    # 숫자 인덱스로 변경
    df = rename_columns_with_index(df)  # (shape: [n_timesteps, n_features_after_renaming])
    # 데이터 값을 0-1 범위로 스케일링
    df = scale_data_to_0_1(df)  # (shape: [n_timesteps, n_features_after_scaling])
    
    return df  # 최종 반환 (shape: [n_timesteps, n_features])

def load_or_generate_missing_data(scaled_df: pd.DataFrame, missing_data_file_path: str, missing_rate: float = 0.1, consecutive_missing_rate: float = 0.1) -> pd.DataFrame:
    """
    결측 데이터 파일을 로드하거나 없을 경우 생성하는 함수

    Args:
        scaled_df (pd.DataFrame): 스케일링된 원본 데이터프레임 (shape: [n_timesteps, n_features])
            - n_timesteps: 타임스텝의 수 (행의 개수)
            - n_features: feature의 개수 (열의 개수)
        missing_data_file_path (str): 결측 데이터 파일 경로
        missing_rate (float): 결측률 (default: 0.1)
        consecutive_missing_rate (float): 연속 결측률 (default: 0.1)

    Returns:
        pd.DataFrame: 결측 데이터프레임 (shape: [n_timesteps, n_features])
            - n_timesteps: 타임스텝의 수 (행의 개수)
            - n_features: feature의 개수 (열의 개수)
    """
    if os.path.exists(missing_data_file_path):
        try:
            # 결측 데이터 파일이 존재하면 'latin1' 인코딩으로 로드
            data_with_missing = pd.read_csv(missing_data_file_path, encoding='latin1')  # (shape: [n_timesteps, n_features])
            print(f"Loaded existing missing data from {missing_data_file_path} with 'latin1' encoding.")
        except UnicodeDecodeError as e:
            # 인코딩 실패 시 'ISO-8859-1' 인코딩으로 다시 시도
            print(f"Error: {e}")
            data_with_missing = pd.read_csv(missing_data_file_path, encoding='ISO-8859-1')  # (shape: [n_timesteps, n_features])
            print(f"Loaded existing missing data from {missing_data_file_path} with 'ISO-8859-1' encoding.")
        
        # 로드한 데이터의 열 이름을 스케일된 데이터와 일치시킴
        data_with_missing.columns = scaled_df.columns  # (shape remains [n_timesteps, n_features])
    else:
        # 결측 데이터 파일이 없으면 데이터를 생성
        data_with_missing = generate_missing_data(scaled_df, missing_rate, consecutive_missing_rate)  # (shape: [n_timesteps, n_features])
        # 생성된 결측 데이터를 지정된 경로에 저장
        save_dataframe_to_csv(data_with_missing, missing_data_file_path)
    
    return data_with_missing  # 최종 반환 (shape: [n_timesteps, n_features])

def remove_first_column_if_timestamp(data: pd.DataFrame) -> pd.DataFrame:
    """
    타임스탬프 열 제거하는 함수

    Args:
        data (pd.DataFrame): time series data 데이터프레임 (shape: [n_timesteps, n_features])
            - n_timesteps: 타임스텝의 수 (행의 개수)
            - n_features: feature의 개수 (열의 개수)

    Returns:
        pd.DataFrame: 타임스탬프 열이 제거된 데이터프레임 (shape: [n_timesteps, n_features - 1] or [n_timesteps, n_features])
            - n_timesteps: 타임스텝의 수 (행의 개수)
            - n_features - 1: 타임스탬프 열이 제거된 경우의 열 수
    """
    if 'Date Time' in data.columns:
        # 'Date Time'이라는 열 이름이 있으면 제거
        data = data.drop(columns=['Date Time'])  # (shape: [n_timesteps, n_features - 1])
        print(f"Date Time column removed.")
    
    # 첫 번째 열 이름을 가져옴
    first_column = data.columns[0]
    try:
        # 첫 번째 열이 타임스탬프 형식인지 확인
        pd.to_datetime(data[first_column])
        # 타임스탬프 형식이면 해당 열을 제거
        data = data.drop(columns=[first_column])  # (shape: [n_timesteps, n_features - 1])
        print(f"First column '{first_column}' recognized as timestamp and removed.")
    except (ValueError, TypeError):
        # 타임스탬프 형식이 아니면 그대로 유지 (shape remains [n_timesteps, n_features])
        print(f"First column '{first_column}' is not a timestamp.")
    
    return data  # 최종 반환 (shape: [n_timesteps, n_features - 1] or [n_timesteps, n_features])

def rename_columns_with_index(data: pd.DataFrame, prefix: str = '') -> pd.DataFrame:
    """
    열 이름을 지우고 열 번호를 기반으로 변경하고 접두사를 추가하는 함수

    Args:
        data (pd.DataFrame): 열 이름을 변경할 데이터프레임 (shape: [n_timesteps, n_features])
            - n_timesteps: 타임스텝의 수 (행의 개수)
            - n_features: feature의 개수 (열의 개수)
        prefix (str): 추가할 접두사 (default: '')

    Returns:
        pd.DataFrame: 열 이름이 열 번호를 기반으로 변경된 데이터프레임 (shape: [n_timesteps, n_features])
            - n_timesteps: 타임스텝의 수 (행의 개수)
            - n_features: feature의 개수 (열의 개수)
    """
    # 열 이름을 숫자 인덱스로 변경하고 접두사를 추가하여 새 열 이름을 생성
    new_columns = {col: f"{prefix}{idx + 1}" for idx, col in enumerate(data.columns)}
    # 새 열 이름으로 데이터프레임의 열 이름을 변경 (shape remains [n_timesteps, n_features])
    return data.rename(columns=new_columns)

def scale_data_to_0_1(data: pd.DataFrame) -> pd.DataFrame:
    """
    데이터를 0에서 1 사이의 값으로 스케일링하는 함수

    Args:
        data (pd.DataFrame): 스케일링할 데이터프레임 (shape: [n_timesteps, n_features])
            - n_timesteps: 타임스텝의 수 (행의 개수)
            - n_features: feature의 개수 (열의 개수)

    Returns:
        pd.DataFrame: 스케일링된 데이터프레임 (shape: [n_timesteps, n_features])
            - n_timesteps: 타임스텝의 수 (행의 개수)
            - n_features: feature의 개수 (열의 개수)
    """
    # MinMaxScaler를 사용하여 데이터를 0-1 범위로 스케일링
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)  # (shape: [n_timesteps, n_features])
    # 스케일링된 데이터를 원본 데이터프레임의 형식으로 반환 (shape: [n_timesteps, n_features])
    scaled_df = pd.DataFrame(scaled_data, columns=data.columns, index=data.index)
    return scaled_df  # 최종 반환 (shape: [n_timesteps, n_features])
