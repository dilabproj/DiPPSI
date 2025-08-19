import pickle
import matplotlib.pyplot as plt
import torch
import numpy as np

# 打开文件，指定二进制读取模式
with open('out/generated_outputs_nsample100.pk', 'rb') as file:
    # 使用pickle.load()函数从文件中加载对象
    loaded_dict = pickle.load(file)

print(loaded_dict[0])
print(loaded_dict[0].shape)
# 输出: {'key': 'value'}

data = loaded_dict[0][:,:].cpu().numpy() # 不是0 不是2 不是3
data_1 = np.load('out/scientisst_ecg_test/csdiv1_scientisst_ecg/original.npy')
data_1 = data_1[0,:,0]

# 绘制图形
plt.figure(figsize=(10, 6))  # 设置图形大小
# for i in range(data.shape[0]):
data[0][data_1 != 0] = np.nan
plt.plot(data[0], label=f'Series {0}')  # 绘制每个序列
plt.plot(data_1, label=f'Series {1}')

plt.title('Data Series')
plt.xlabel('Index')
plt.ylabel('Value')
plt.legend()  # 添加图例
plt.show()
# 保存图形到文件
plt.savefig('data_series_plot.png', dpi=300)  # 指定文件名和分辨率
