import os
import glob
import scipy.io
import numpy as np
import torch
from tqdm import tqdm

def add_transient_missingness(X, impute_transient):
    """
    Adds transient missingness to the data based on impute_transient settings.
    """
    input = X.clone()  # Clone the original data to create a modified version
    target = torch.full_like(X, float('nan'))  # Initialize target with NaNs

    total_len = X.shape[1]
    # Extract settings from impute_transient
    amt_impute = impute_transient["window"]
    prob_missing = impute_transient["prob"]

    for i in range(X.shape[0]):  # iterating through all data points
        for start_impute in range(0, total_len, amt_impute):
            for j in range(X.shape[-1]):
                rand = np.random.random_sample()
                if rand < prob_missing:
                    input[i, start_impute:start_impute+amt_impute, j] = np.nan
                    # For target, replace NaNs with original data where missingness is added
                    target[i, start_impute:start_impute+amt_impute, j] = X[i, start_impute:start_impute+amt_impute, j]
                    X[i, start_impute:start_impute+amt_impute, j] = 0

    return X, input, target


def load(mean=None, mode=True, bounds=1, impute_transient=None, channels=['ppg:wrist', 'acc_e4:z', 'acc_e4:x', 'acc_e4:y'], train=True, val=True, test=False, addmissing=True, path="/mnt/1stHDD/nfs/roeywu/PPG_data/PPG_ACC_dataset/S1"):
    segment_data = []  # 用来收集所有数据段
    segment_lengths = []  # 记录每个数据段的长度，以便后续分割

    sampling_rate = 64  # 64 Hz
    segment_length = 5 * 60 * sampling_rate  # 5 minutes in samples

    # 使用 glob 查找所有 _ppg.mat 文件
    mat_files = glob.glob(os.path.join(path, '*_ppg.mat'))
    print(f"Found {len(mat_files)} .mat files.")
    print("mat_files: ", mat_files)

    for filename in tqdm(mat_files):
        try:
            # 加载 .mat 文件
            data_dict = scipy.io.loadmat(filename)
            data = data_dict['PPG'][:,1]  # 假设数据在 'ppg_data' 键中

            # Process each segment of 5 minutes, collect segments
            for start_idx in range(0, data.shape[0], segment_length):
                end_idx = start_idx + segment_length
                if end_idx > data.shape[0]:
                    break  # Discard the remainder that doesn't form a full 5-minute segment

                segment = data[start_idx:end_idx]
                segment_data.append(segment)
                segment_lengths.append(len(segment))

        except KeyError:
            print(f"No 'PPG' in file {filename}, skipping.")
        except Exception as e:
            print(f"Error processing file {filename}: {str(e)}")

    # 使用 np.vstack 直接垂直堆叠这些数组
    concatenated_segments = np.vstack(segment_data)

    # Further processing and return
    concatenated_segments = torch.from_numpy(concatenated_segments).float()  # 确保使用.float()来转换为 FloatTensor
    concatenated_segments = concatenated_segments.unsqueeze(2)

    # 初始化目标张量，同样确保为浮点型
    target = torch.full_like(concatenated_segments, float('nan'), dtype=torch.float32)  # 使用dtype指定类型
    print(concatenated_segments.shape)
    if addmissing:
        test_data, test_input, test_target = add_transient_missingness(concatenated_segments, impute_transient)
    else:
        modified_data = concatenated_segments  # No missingness added

    # Construct Y_dict which might include target and any other labels or additional information
    Y_dict = {"target_seq": test_target}
    return test_data, Y_dict

# Assume we have a function similar to modify that accepts EDF data and does something
def modify(X, type=None, addmissing=False):
    target = np.empty(X.shape, dtype=np.float32)
    target[:] = np.nan
    target = torch.from_numpy(target)
    
    if addmissing:
        print("adding missing")
        miss_tuples_path = os.path.join("/mnt/1stHDD/nfs/roeywu/PPG_data","missingness_patterns/mHealth_missing_ppg", f"missing_ppg_{type}.csv")
        with open(miss_tuples_path, 'r') as read_obj:
            csv_reader = reader(read_obj)
            list_of_miss = list(csv_reader)
        # X is 15174, bf is 43
        print(len(list_of_miss))
        for iter_idx, waveform_idx in enumerate(tqdm(range(0, X.shape[0], 4))):
            ppgmiss_idx = iter_idx % len(list_of_miss)
            miss_vector = miss_tuple_to_vector(list_of_miss[ppgmiss_idx])
            print("miss_vector.shape: ", miss_vector.shape)
            if X.shape[0] - waveform_idx < 4:
                totalrange = X.shape[0] - waveform_idx 
            else:
                totalrange = 4
            for i in range(totalrange): # bs of 4, for that batch, we have the same missingness pattern
                target[waveform_idx + i, np.where(miss_vector == 0)[0]] = X[waveform_idx + i, np.where(miss_vector == 0)[0]]
                X[waveform_idx + i, :, :] = X[waveform_idx + i, :, :] * miss_vector
    return X, {"target_seq":target}

def miss_tuple_to_vector(listoftuples):
    def onesorzeros_vector(miss_tuple):
        print(type(miss_tuple))
        miss_tuple = literal_eval(miss_tuple)
        print("miss_tuple: ", miss_tuple)
        if miss_tuple[0] == 0:
            return np.zeros(miss_tuple[1])
        elif miss_tuple[0] == 1:
            return np.ones(miss_tuple[1])

    miss_vector = onesorzeros_vector(listoftuples[0])
    print(len(listoftuples))
    print(miss_vector.shape)
    for i in range(1, len(listoftuples)):
        miss_vector = np.concatenate((miss_vector, onesorzeros_vector(listoftuples[i])))
        print(miss_vector.shape)
    miss_vector = np.expand_dims(miss_vector, 1)
    return miss_vector
