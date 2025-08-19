import numpy as np

# 指定NumPy檔案的路徑
file_path = "/mnt/1stHDD/nfs/roeywu/PPG_data/data/mimic_ppg/mimic_ppg_test.npy"
file_path2 = "/mnt/2ndHDD/nfs/roeywu/PPG_data/data/pulseimpute_data/waveforms/mimic_ppg/mimic_ppg_test.npy"
# 使用numpy加載檔案
data = np.load(file_path)
data2 = np.load(file_path2)
# 打印檔案維度
print(f"The dimensions of the file '{file_path}' are: {data.shape}")
print(f"The dimensions of the file '{file_path2}' are: {data2.shape}")
# 打印內容
print(data[0])
