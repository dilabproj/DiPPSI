import os
import numpy as np
#import ipympl
import mne
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
from configs.test_transformer_configs import *
from configs.test_brits_configs import *
from configs.test_other_configs import *
from configs.test_naomi_configs import *

from utils.evaluate_imputation import eval_mse, eval_heartbeat_detection, eval_cardiac_classification, printlog
import numpy as np
import torch
import random
from tqdm import tqdm
from csv import reader
from ast import literal_eval

# def random_seed(seed_value, use_cuda):
#     np.random.seed(seed_value) # cpu vars
#     torch.manual_seed(seed_value) # cpu  vars
#     random.seed(seed_value) # Python
#     if use_cuda: 
#         torch.cuda.manual_seed(seed_value)
#         torch.cuda.manual_seed_all(seed_value) # gpu vars

bootstrap = (1000, 1) # num of bootstraps, size of bootstrap sample compared to test size
configs = [lininterp_scientisst_ecg_testtransient_30percent, mean_scientisst_ecg_testtransient_30percent]


# def load(mean=None, bounds=None, train=True, val=True, test=False, addmissing=False, path=os.path.join("/mnt/1stHDD/nfs/roeywu/PPG_data/data/mimic_ppg")):

#     # Train
#     if train:
#         X_train = np.load(os.path.join(path, "mimic_ppg_train.npy")).astype(np.float32)
#         if mean:
#             X_train -= np.mean(X_train,axis=1,keepdims=True) 
#         if bounds:
#             X_train /= np.amax(np.abs(X_train),axis=1,keepdims=True)*bounds
#         X_train, Y_dict_train = modify(X_train, type="train", addmissing=False)
#     else:
#         X_train = None
#         Y_dict_train = None

#     # Val
#     if val:
#         X_val = np.load(os.path.join(path, "mimic_ppg_val.npy")).astype(np.float32)
#         if mean:
#             X_val -= np.mean(X_val,axis=1,keepdims=True) 
#         if bounds:
#             X_val /= np.amax(np.abs(X_val),axis=1,keepdims=True)*bounds
#         X_val, Y_dict_val = modify(X_val, type="val", addmissing=False)
#     else:
#         X_val = None
#         Y_dict_val = None

#     # Test
#     if test:
#         X_test = np.load(os.path.join(path, "mimic_ppg_test.npy")).astype(np.float32)
#         if mean:
#             X_test -= np.mean(X_test,axis=1,keepdims=True) 
#         if bounds:
#             X_test /= np.amax(np.abs(X_test),axis=1,keepdims=True)*bounds
#         X_test, Y_dict_test = modify(X_test, type="test", addmissing=addmissing)
#     else:
#         X_test = None
#         Y_dict_test = None

#     return X_train, Y_dict_train, X_val, Y_dict_val, X_test, Y_dict_test

# def modify(X, type=None, addmissing=False):
#     X = torch.from_numpy(X)
#     target = np.empty(X.shape, dtype=np.float32)
#     target[:] = np.nan
#     target = torch.from_numpy(target)

#     if addmissing:
#         print("adding missing")
#         miss_tuples_path = os.path.join("/mnt/1stHDD/nfs/roeywu/PPG_data","missingness_patterns/mHealth_missing_ppg", f"missing_ppg_{type}.csv")
#         with open(miss_tuples_path, 'r') as read_obj:
#             csv_reader = reader(read_obj)
#             list_of_miss = list(csv_reader)
#         # X is 15174, bf is 43
#         for iter_idx, waveform_idx in enumerate(tqdm(range(0, X.shape[0], 4))):
#             ppgmiss_idx = iter_idx % len(list_of_miss)
#             miss_vector = miss_tuple_to_vector(list_of_miss[ppgmiss_idx])
#             if X.shape[0] - waveform_idx < 4:
#                 totalrange = X.shape[0] - waveform_idx 
#             else:
#                 totalrange = 4
#             for i in range(totalrange): # bs of 4, for that batch, we have the same missingness pattern
#                 target[waveform_idx + i, np.where(miss_vector == 0)[0]] = X[waveform_idx + i, np.where(miss_vector == 0)[0]]
#                 X[waveform_idx + i, :, :] = X[waveform_idx + i, :, :] * miss_vector
#     return X, {"target_seq":target}

# def miss_tuple_to_vector(listoftuples):
#     def onesorzeros_vector(miss_tuple):
#         miss_tuple = literal_eval(miss_tuple)
#         if miss_tuple[0] == 0:
#             return np.zeros(miss_tuple[1])
#         elif miss_tuple[0] == 1:
#             return np.ones(miss_tuple[1])

#     miss_vector = onesorzeros_vector(listoftuples[0])
#     for i in range(1, len(listoftuples)):
#         miss_vector = np.concatenate((miss_vector, onesorzeros_vector(listoftuples[i])))
#     miss_vector = np.expand_dims(miss_vector, 1)
#     return miss_vector

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


def load(mean=None, mode=True, bounds=1, impute_transient=None, channels=['ecg:dry', 'acc_chest:z', 'acc_chest:x', 'acc_chest:y'], train=True, val=True, test=False, addmissing=False, path="/mnt/1stHDD/nfs/roeywu/PPG_data/physionet.org/files/scientisst-move-biosignals/1.0.0"):
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

        filename = os.path.join(path, patient_id, 'scientisst_chest.edf')
        try:
            raw = mne.io.read_raw_edf(filename, preload=True)
        except FileNotFoundError:
            print(f"File not found for patient {patient_id}, skipping.")
            continue

        # Indices for accelerometer data
        ecg_indices = [raw.ch_names.index(channel) for channel in channels if channel in raw.ch_names]

        if not ecg_indices:
            print(f"No accelerometer data found for patient {patient_id}.")
            continue

        data = raw.get_data(picks=ecg_indices).astype(np.float32)

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
    if addmissing:
        modified_data, input_with_missingness = add_transient_missingness(concatenated_segments, impute_transient)
        # Fill in the target with the original data where the missingness will be added
        for i in range(concatenated_segments.shape[0]):  # iterate over all data points
            for j in range(concatenated_segments.shape[-1]):  # iterate over all channels
                missing_indices = torch.isnan(input_with_missingness[i, :, j])
                target[i, missing_indices, j] = concatenated_segments[i, missing_indices, j]
    else:
        modified_data = concatenated_segments  # No missingness added
        target = concatenated_segments

    # Construct Y_dict which might include target and any other labels or additional information
    Y_dict = {"target_seq": target}
    # Add other necessary items to Y_dict as required by your project

    return modified_data, Y_dict

# X_train, Y_dict_train, X_val, Y_dict_val, X_test, Y_dict_test = load(**configs[0]["data_load"])
X_test, Y_dict_test = load(**configs[0]["data_load"])

# print("X_train: ", X_train)
# print("Y_dict_train: ", Y_dict_train)
# print("X_val: ", X_val)
# print("Y_dict_val: ", Y_dict_val)
# print("X_test.shape: ", X_test.shape)
# print("Y_dict_test: ", Y_dict_test)

# INIT
out_folder = "out"
tasks = sorted([task for task in os.listdir(out_folder) if not task.startswith('.')])  # Deal with /.DS_Store
print(tasks)
current_task = tasks[17]
current_models = sorted([task for task in os.listdir(os.path.join(out_folder, current_task)) if not task.startswith('.')])
print(current_models)
current_sample_index = 0
channels = [0]

# LOAD
def load_data(task, model):
    task_path = os.path.join(out_folder, task, model)
    original_data = np.load(os.path.join(task_path, "original.npy"))
    original_data = original_data[:,:,channels]
    imputation_data = np.load(os.path.join(task_path, "imputation.npy"))
    target_seq_data = Y_dict_test["target_seq"]
    return original_data, imputation_data, target_seq_data

default_sample_length = len(load_data(current_task, current_models[1])[0][current_sample_index])

# main fn
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio

impute_extended = 100
impute_transient = None

def update_plot(task, sample_index):
    original_data_fft, imputation_data_fft, target_seq_data_fft = load_data(task, current_models[1])
    original_data_lininterp, imputation_data_lininterp, target_seq_data_lininterp = load_data(task, current_models[2])
    # 检查 imputation_data 是否包含 0 值
    if np.any(imputation_data_lininterp == 0):
        print("imputation_data_lininterp contains zero values.")
    else:
        print("imputation_data_lininterp does not contain any zero values.")

    original_data_mean, imputation_data_mean, target_seq_data_mean = load_data(task, current_models[3])
    original_data_naomi, imputation_data_naomi, target_seq_data_naomi = load_data(task, current_models[4])

    # Get the length of the current sample
    sample_length = len(original_data_lininterp[sample_index])
    print(sample_length)
    print(len(original_data_naomi[sample_index]))
    print("sample_length: ", sample_length)

    x_range = (0,sample_length)

    # # Convert zero values to NaN
    # imputation_data_fft[sample_index][original_data_fft[sample_index] != 0] = np.nan
    # original_data_fft[sample_index][original_data_fft[sample_index] == 0] = np.nan
    # target_seq_data_fft[sample_index][target_seq_data_fft[sample_index] == 0] = np.nan

    # isnan = torch.isnan(original_data_fft[sample_index][:sample_length//30]).any()
    # print(f"Data contains NaN: {isnan}")

    # if isnan:
    #     data_without_nan = torch.nan_to_num(original_data_fft[sample_index][:sample_length//30], nan=0.0)
    # else:
    #     data_without_nan = original_data_fft[sample_index][:sample_length//30]

    # min_val, _ = torch.min(data_without_nan, axis=0)
    # max_val, _ = torch.max(data_without_nan, axis=0)
    # min_val_value = min_val.item()
    # max_val_value = max_val.item()

    # # 使用提取的最小值和最大值进行归一化
    # min_max_normalized_data = (original_data_fft[sample_index][:sample_length//30] - min_val_value) / (max_val_value - min_val_value)

    # Convert zero values to NaN
    imputation_data_lininterp[sample_index][original_data_lininterp[sample_index] != 0] = np.nan
    original_data_lininterp[sample_index][original_data_lininterp[sample_index] == 0] = np.nan

    # Convert zero values to NaN
    imputation_data_mean[sample_index][original_data_mean[sample_index] != 0] = np.nan
    original_data_mean[sample_index][original_data_mean[sample_index] == 0] = np.nan

    # Convert zero values to NaN
    imputation_data_fft[sample_index][original_data_fft[sample_index] != 0] = np.nan
    original_data_fft[sample_index][original_data_fft[sample_index] == 0] = np.nan

    # # Convert zero values to NaN
    imputation_data_naomi[sample_index][original_data_naomi[sample_index] != 0] = np.nan
    original_data_naomi[sample_index][original_data_naomi[sample_index] == 0] = np.nan

    # Create the plot
    fig = make_subplots(rows=4, cols=1)

    fig.add_trace(go.Scatter(x=np.arange(sample_length//10),
                             y=target_seq_data_fft[sample_index][:sample_length//10].flatten(),
                             mode='lines',
                             name='FFT Target Sequence',
                             line=dict(color='lightgray'),
                             ), row=1, col=1)

    fig.add_trace(go.Scatter(x=np.arange(sample_length//10),
                             y=original_data_fft[sample_index][:sample_length//10].flatten(),
                             mode='lines',
                             name='FFT Original',
                             line=dict(color='black'),
                             ), row=1, col=1)

    fig.add_trace(go.Scatter(x=np.arange(sample_length//10),
                             y=imputation_data_fft[sample_index][:sample_length//10].flatten(),
                             mode='lines',
                             name='FFT Imputation',
                             line=dict(color='dodgerblue', width=2),
                             ), row=1, col=1)

    fig.add_trace(go.Scatter(x=np.arange(sample_length//10),
                             y=target_seq_data_lininterp[sample_index][:sample_length//10].flatten(),
                             mode='lines',
                             name='Lininterp Target Sequence',
                             line=dict(color='lightgray'),
                             ), row=2, col=1)

    fig.add_trace(go.Scatter(x=np.arange(sample_length//10),
                             y=original_data_lininterp[sample_index][:sample_length//10].flatten(),
                             mode='lines',
                             name='Lininterp Original',
                             line=dict(color='black'),
                             ), row=2, col=1)

    fig.add_trace(go.Scatter(x=np.arange(sample_length//10),
                             y=imputation_data_lininterp[sample_index][:sample_length//10].flatten(),
                             mode='lines',
                             name='Lininterp Imputation',
                             line=dict(color='dodgerblue', width=2),
                             ), row=2, col=1)

    fig.add_trace(go.Scatter(x=np.arange(sample_length//10),
                             y=target_seq_data_mean[sample_index][:sample_length//10].flatten(),
                             mode='lines',
                             name='Mean Target Sequence',
                             line=dict(color='lightgray'),
                             ), row=3, col=1)

    fig.add_trace(go.Scatter(x=np.arange(sample_length//10),
                             y=original_data_mean[sample_index][:sample_length//10].flatten(),
                             mode='lines',
                             name='Mean Original',
                             line=dict(color='black'),
                             ), row=3, col=1)

    fig.add_trace(go.Scatter(x=np.arange(sample_length//10),
                             y=imputation_data_mean[sample_index][:sample_length//10].flatten(),
                             mode='lines',
                             name='Mean Imputation',
                             line=dict(color='dodgerblue', width=2),
                             ), row=3, col=1)

    fig.add_trace(go.Scatter(x=np.arange(sample_length//10),
                             y=target_seq_data_naomi[sample_index][:sample_length//10].flatten(),
                             mode='lines',
                             name='Naomi step256 Target Sequence',
                             line=dict(color='lightgray'),
                             ), row=4, col=1)

    fig.add_trace(go.Scatter(x=np.arange(sample_length//10),
                             y=original_data_naomi[sample_index][:sample_length//10].flatten(),
                             mode='lines',
                             name='Naomi step256 Original',
                             line=dict(color='black'),
                             ), row=4, col=1)

    fig.add_trace(go.Scatter(x=np.arange(sample_length//10),
                             y=imputation_data_naomi[sample_index][:sample_length//10].flatten(),
                             mode='lines',
                             name='Naomi step256 Imputation',
                             line=dict(color='dodgerblue', width=2),
                             ), row=4, col=1)

    # 更新每个子图的X轴和Y轴，为它们设置标签
    fig.update_xaxes(title_text="Data Points", row=1, col=1)
    fig.update_yaxes(title_text="Values", row=1, col=1)

    fig.update_xaxes(title_text="Data Points", row=2, col=1)
    fig.update_yaxes(title_text="Values", row=2, col=1)

    fig.update_xaxes(title_text="Data Points", row=3, col=1)
    fig.update_yaxes(title_text="Values", row=3, col=1)

    fig.update_xaxes(title_text="Data Points", row=4, col=1)
    fig.update_yaxes(title_text="Values", row=4, col=1)

    fig.update_layout(
        title="Imputation Visualization",
        height=1200,  # Set the height of the plot
        width=800,    # Set the width of the plot
        plot_bgcolor='white'
    )

    # Convert the plot to an image
    img_bytes = pio.to_image(fig, format='png')

    # Construct the file path for saving the image
    save_path = os.path.join(out_folder, task, f"imputation_visualization_{sample_index}.png")
    
    # Save the plot as a PNG file
    pio.write_image(fig, save_path)

    print(f"Plot saved as {save_path}")

update_plot(current_task, current_sample_index)
