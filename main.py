import os
import yaml
import sys
from cubicspline_interpolation.cubicspline_interpolation import CubicSplineInterpolation
from focalizeKnn_imputation.focalizeKnn_imputation import FocalizeKNNImputation
from fusion_imputation.fusion_imputation import FusionImputation
from Knn_imputation.Knn_Imputation import KNNImputation
from knndtw_imputation.knndtw_imputation import KNNDTWImputation
from linear_interpolation.linear_interpoaltion import LinearInterpolation
from movingAverage_interpolation.movingAverage_Interpolation import MovingAverageInterpolation
from nearest_interpolation.neareast_interpolation import NearestInterpolation

def load_config(method):
    # Load common config
    common_config_path = 'yaml_file/config_common.yaml'
    try:
        with open(common_config_path, 'r', encoding='utf-8') as file:
            common_config = yaml.safe_load(file)
        if common_config is None:
            raise ValueError(f"Common config file is empty: {common_config_path}")
    except FileNotFoundError:
        print(f"Common config file not found: {common_config_path}")
        return None
    except yaml.YAMLError as e:
        print(f"Error parsing common config file: {e}")
        return None
    except UnicodeDecodeError as e:
        print(f"Encoding error when reading {common_config_path}: {e}")
        return None
    
    # Load method-specific config
    method_config_path = f'yaml_file/config_{method}.yaml'
    try:
        with open(method_config_path, 'r', encoding='utf-8') as file:
            method_config = yaml.safe_load(file)
        if method_config is None:
            method_config = {}  # Use empty dict if file is empty
    except FileNotFoundError:
        print(f"Method config file not found: {method_config_path}")
        return None
    except yaml.YAMLError as e:
        print(f"Error parsing method config file: {e}")
        return None
    except UnicodeDecodeError as e:
        print(f"Encoding error when reading {method_config_path}: {e}")
        return None
    
    # Merge configs, method-specific config takes precedence
    config = {**common_config, **method_config}
    return config



def main(method):
    config = load_config(method)
    if config is None:
        print(f"Failed to load configuration for method: {method}")
        return

    # 수정된 부분
    output_dir = f"{config['str_file']}all_results"
    os.makedirs(output_dir, exist_ok=True)

    if method == 'linear':
        imputer = LinearInterpolation(
            data_file_path=config['data_path'],
            missing_data_file_path=config['missing_data_path'],
            output_dir=os.path.join(output_dir, f"{config['str_file']}linear_result")
        )
    elif method == 'nearest':
        imputer = NearestInterpolation(
            data_file_path=config['data_path'],
            missing_data_file_path=config['missing_data_path'],
            output_dir=os.path.join(output_dir, f"{config['str_file']}nearest_result")
        )
    elif method == 'moving_average':
        imputer = MovingAverageInterpolation(
            data_file_path=config['data_path'],
            missing_data_file_path=config['missing_data_path'],
            output_dir=os.path.join(output_dir, f"{config['str_file']}MovingAvg_result"),
            window=config['window']
        )
    elif method == 'knn':
        imputer = KNNImputation(
            data_file_path=config['data_path'],
            missing_data_file_path=config['missing_data_path'],
            output_dir=os.path.join(output_dir, f"{config['str_file']}knn_result"),
            n_neighbors=config['n_neighbors']
        )
    elif method == 'knn_dtw':
        imputer = KNNDTWImputation(
            data_file_path=config['data_path'],
            missing_data_file_path=config['missing_data_path'],
            output_dir=os.path.join(output_dir, f"{config['str_file']}knn_dtw_result"),
            n_neighbors=config['n_neighbors']
        )
    elif method == 'focalize_knn':
        imputer = FocalizeKNNImputation(
            data_file_path=config['data_path'],
            missing_data_file_path=config['missing_data_path'],
            output_dir=os.path.join(output_dir, f"{config['str_file']}FkNN_result"),
            n_neighbors=config['n_neighbors'],
            threshold=config['threshold'],
            lags=config['lags']
        )
    elif method == 'cubic_spline':
        imputer = CubicSplineInterpolation(
            data_file_path=config['data_path'],
            missing_data_file_path=config['missing_data_path'],
            output_dir=os.path.join(output_dir, f"{config['str_file']}cubic_spline_result")
        )
    elif method == 'fusion':
        imputer = FusionImputation(
            num_closest_features=config['num_closest_features'],
            num_similar_segments=config['num_similar_segments'],
            dtw_max_warping_segment=config['dtw_max_warping_segment'],
            msm_cost=config['msm_cost'],
            segment_step_size=config['segment_step_size'],
            segment_random_ratio=config['segment_random_ratio'],
            spearman_exponent=config['spearman_exponent'],
            output_dir=os.path.join(output_dir, f"{config['str_file']}Fusion_MTSI_result"),
            data_file_path=config['data_path'],
            missing_data_file_path=config['missing_data_path']
        )
    else:
        raise ValueError(f"Unknown method: {method}")

    imputer.run()

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python main.py <method>")
        sys.exit(1)
    method = sys.argv[1]
    main(method)