import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FixedLocator

# 使用索引作为 x 轴数据
x_1 = np.arange(4)  # [0, 1, 2, 3]
labels_1 = ['Conservative', 'Normal', 'Mixed', 'Aggressive']

# 示例数据（第一个子图的数据）
y_TSTF_1 = np.array([0.8581, 0.8371, 0.8242, 0.8096])
y_CoPO_1 = np.array([0.8145, 0.7940, 0.7598, 0.7289])
y_IPPO_1 = np.array([0.7744, 0.7617, 0.7448, 0.7062])
error_TSTF_1 = np.array([0.0124, 0.0132, 0.0119, 0.0217])
error_CoPO_1 = np.array([0.0260, 0.0231, 0.0217, 0.0246])
error_IPPO_1 = np.array([0.0154, 0.0204, 0.0239, 0.0344])

x_2 = np.arange(4)  # [0, 1, 2, 3]
# 将标签换行，避免重叠，并调整字体大小
labels_2 = ['10HVs\n&10AVs', '10HVs\n&15AVs', '15HVs\n&10AVs', '15HVs\n&15AVs']

# 示例数据（第二个子图的数据）
y_TSTF_2 = np.array([0.8371, 0.7926, 0.7705, 0.7019])
y_CoPO_2 = np.array([0.7940, 0.7507, 0.6823, 0.6011])
y_IPPO_2 = np.array([0.7617, 0.7052, 0.6309, 0.5791])
error_TSTF_2 = np.array([0.0132, 0.0109, 0.0180, 0.0195])
error_CoPO_2 = np.array([0.0231, 0.0289, 0.0349, 0.0230])
error_IPPO_2 = np.array([0.0204, 0.0238, 0.0325, 0.0227])

# 设置全局字体
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 45

# 创建包含两个子图的 Figure（横向两个子图）
fig, axs = plt.subplots(1, 2, figsize=(24, 12))

# 第一子图：使用三原色绘制数据
# 使用 Red、Green、Blue 分别表示 TSTF、CoPO、IPPO
axs[0].errorbar(x_1, y_TSTF_1, yerr=error_TSTF_1, label='TSTF', fmt='-o', color='black', capsize=5)
axs[0].errorbar(x_1, y_CoPO_1, yerr=error_CoPO_1, label='CoPO', fmt='-o', color='red', capsize=5)
axs[0].errorbar(x_1, y_IPPO_1, yerr=error_IPPO_1, label='IPPO', fmt='-o', color='blue', capsize=5)
axs[0].set_xticks(x_1)
axs[0].set_xticklabels(labels_1)
axs[0].set_xlabel('Driving Styles of HVs', fontsize=40)
axs[0].set_ylabel('Success', fontsize=40)
axs[0].tick_params(axis='x', labelsize=35)
axs[0].tick_params(axis='y', labelsize=35)
# 图例放在左下角
axs[0].legend(loc="lower left", bbox_to_anchor=(0, 0), borderaxespad=0.1,
              frameon=False, fontsize=35)
# 在 y=0.8 和 y=0.7 处画虚线
# axs[0].axhline(y=0.8, linestyle='--', color='gray', linewidth=2)
# axs[0].axhline(y=0.7, linestyle='--', color='gray', linewidth=2)

# 第二子图：使用三原色绘制数据
axs[1].errorbar(x_2, y_TSTF_2, yerr=error_TSTF_2, label='TSTF', fmt='-o', color='black', capsize=5)
axs[1].errorbar(x_2, y_CoPO_2, yerr=error_CoPO_2, label='CoPO', fmt='-o', color='red', capsize=5)
axs[1].errorbar(x_2, y_IPPO_2, yerr=error_IPPO_2, label='IPPO', fmt='-o', color='blue', capsize=5)
axs[1].set_xticks(x_2)
axs[1].set_xticklabels(labels_2, fontsize=40)
axs[1].set_xlabel('Number of Vehicles', fontsize=40)
axs[1].set_ylabel('Success', fontsize=40)
axs[1].tick_params(axis='x', labelsize=35)
axs[1].tick_params(axis='y', labelsize=35)
axs[1].legend(loc="lower left", bbox_to_anchor=(0, 0), borderaxespad=0.1,
              frameon=False, fontsize=35)
# 在 y=0.8 和 y=0.7 处画虚线
# axs[1].axhline(y=0.8, linestyle='--', color='gray', linewidth=2)
# axs[1].axhline(y=0.7, linestyle='--', color='gray', linewidth=2)

# 调整子图上下间距，使图形与上下边缘间隔更大
plt.subplots_adjust(top=0.98, bottom=0.05, wspace=0.3)

# 保存为 PDF 文件（300 dpi 高质量）
plt.savefig('generalization.pdf', format='pdf', dpi=300, bbox_inches="tight")

# 显示图表
plt.show()
