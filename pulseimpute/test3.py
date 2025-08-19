import scipy.io

# 文件路径
file_path = '/mnt/1stHDD/nfs/roeywu/PPG_data/PPG_ACC_dataset/S1/step1_ppg.mat'

# 使用 scipy.io.loadmat 加载 .mat 文件
data = scipy.io.loadmat(file_path)

# data 现在包含了 .mat 文件中的所有变量
# 可以通过变量名来访问对应的数据，例如：
# ppg_data = data['variable_name_here']

# 打印 data 的键，这通常对应于 MATLAB 中的变量名
print(data.keys())

# 假设我们知道了具体的变量名，可以这样访问它们
# 这里假设 'ppg_signal' 是其中一个存储在 .mat 文件中的变量名
if 'PPG' in data:
    ppg_signal = data['PPG']
    print(ppg_signal)
    print(ppg_signal.shape)
else:
    print("Variable 'ppg_signal' not found in the file")
