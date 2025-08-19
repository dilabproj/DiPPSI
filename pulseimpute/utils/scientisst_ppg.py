import os
import mne
import numpy as np
import torch
from tqdm import tqdm
from csv import reader
from ast import literal_eval

import os
import mne
import numpy as np
import torch
from tqdm import tqdm

def add_transient_missingness(X, impute_transient):
    """
    Adds transient missingness to the data based on impute_transient settings.
    """
    input = X.clone()  # Clone the original data to create a modified version
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
                    X[i, start_impute:start_impute+amt_impute, j] = 0

    return X, input


def load(mean=None, mode=True, bounds=1, impute_transient=None, channels=['ppg:wrist', 'acc_e4:z', 'acc_e4:x', 'acc_e4:y'], train=True, val=True, test=False, addmissing=False, path="/mnt/1stHDD/nfs/roeywu/PPG_data/physionet.org/files/scientisst-move-biosignals/1.0.0"):
    segment_data = []  # 用來收集所有病患的數據段
    segment_lengths = []  # 記錄每個數據段的長度，以便後續分割

    sampling_rate = 64  # 64 Hz
    segment_length = 5 * 60 * sampling_rate  # 5 minutes in samples

    patient_ids = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
    print(f"Found {len(patient_ids)} patients.")

    for patient_id in tqdm(patient_ids):
        if patient_id == "93JD":
            print("Skipping patient 93JD.")
            continue

        filename = os.path.join(path, patient_id, 'empatica.edf')
        try:
            raw = mne.io.read_raw_edf(filename, preload=True)
        except FileNotFoundError:
            print(f"File not found for patient {patient_id}, skipping.")
            continue

        ppg_indices = [raw.ch_names.index(channel) for channel in channels if channel in raw.ch_names]

        if not ppg_indices:
            print(f"No accelerometer data found for patient {patient_id}.")
            continue

        data = raw.get_data(picks=ppg_indices).astype(np.float32)

        # Process each segment of 5 minutes, collect segments
        for start_idx in range(0, data.shape[1], segment_length):
            end_idx = start_idx + segment_length
            if end_idx > data.shape[1]:
                break  # Discard the remainder that doesn't form a full 5-minute segment

            segment = data[:, start_idx:end_idx]
            segment_data.append(segment)
            segment_lengths.append(segment.shape[1])

    # 使用 np.vstack 直接垂直堆叠这些数组
    concatenated_segments = np.vstack(segment_data)
    # 再次检查合并后的数组形状
    print(concatenated_segments.shape)

    # 确保没有触发 IndexError
    try:
        print(concatenated_segments.shape[0], concatenated_segments.shape[1])
    except IndexError as e:
        print("Error:", e)

    # Apply modifications
    concatenated_segments = torch.from_numpy(concatenated_segments)
    concatenated_segments = concatenated_segments.unsqueeze(2)
    target = torch.full_like(concatenated_segments, float('nan'))  # Initialize target with nans
    print(concatenated_segments.shape)
    if addmissing:
        modified_data, input_with_missingness = add_transient_missingness(concatenated_segments, impute_transient)
        # Fill in the target with the original data where the missingness will be added
        for i in range(concatenated_segments.shape[0]):  # iterate over all data points
            for j in range(concatenated_segments.shape[-1]):  # iterate over all channels
                missing_indices = torch.isnan(input_with_missingness[i, :, j])
                target[i, missing_indices, j] = concatenated_segments[i, missing_indices, j]
    else:
        modified_data = concatenated_segments  # No missingness added

    # Construct Y_dict which might include target and any other labels or additional information
    Y_dict = {"target_seq": target}
    # Add other necessary items to Y_dict as required by your project

    return modified_data, Y_dict

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
