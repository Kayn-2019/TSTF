import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 读取每个算法的三组CSV文件，分别读取timesteps_total和success, episode_reward_mean, crash列
df_algo1_group1 = pd.read_csv('progress_ippo_nor_1.csv',
                              usecols=['timesteps_total', 'success', 'episode_reward_mean', 'crash', 'out'])
df_algo1_group2 = pd.read_csv('progress_ippo_nor_2.csv',
                              usecols=['timesteps_total', 'success', 'episode_reward_mean', 'crash', 'out'])
df_algo1_group3 = pd.read_csv('progress_ippo_nor_3.csv',
                              usecols=['timesteps_total', 'success', 'episode_reward_mean', 'crash', 'out'])

df_algo2_group1 = pd.read_csv('progress_copo_nor_1.csv',
                              usecols=['timesteps_total', 'success', 'episode_reward_mean', 'crash', 'out'])
df_algo2_group2 = pd.read_csv('progress_copo_nor_2.csv',
                              usecols=['timesteps_total', 'success', 'episode_reward_mean', 'crash', 'out'])
df_algo2_group3 = pd.read_csv('progress_copo_nor_3.csv',
                              usecols=['timesteps_total', 'success', 'episode_reward_mean', 'crash', 'out'])

df_algo3_group1 = pd.read_csv('progress_ccpo_mf_nor_1.csv',
                              usecols=['timesteps_total', 'success', 'episode_reward_mean', 'crash', 'out'])
df_algo3_group2 = pd.read_csv('progress_ccpo_mf_nor_2.csv',
                              usecols=['timesteps_total', 'success', 'episode_reward_mean', 'crash', 'out'])
df_algo3_group3 = pd.read_csv('progress_ccpo_mf_nor_3.csv',
                              usecols=['timesteps_total', 'success', 'episode_reward_mean', 'crash', 'out'])

df_algo4_group1 = pd.read_csv('progress_ccpo_concat_nor_1.csv',
                              usecols=['timesteps_total', 'success', 'episode_reward_mean', 'crash', 'out'])
df_algo4_group2 = pd.read_csv('progress_ccpo_concat_nor_2.csv',
                              usecols=['timesteps_total', 'success', 'episode_reward_mean', 'crash', 'out'])
df_algo4_group3 = pd.read_csv('progress_ccpo_concat_nor_3.csv',
                              usecols=['timesteps_total', 'success', 'episode_reward_mean', 'crash', 'out'])

df_algo5_group1 = pd.read_csv('progress_tstf_nor_1.csv',
                              usecols=['timesteps_total', 'success', 'episode_reward_mean', 'crash', 'out'])
df_algo5_group2 = pd.read_csv('progress_tstf_nor_2.csv',
                              usecols=['timesteps_total', 'success', 'episode_reward_mean', 'crash', 'out'])
df_algo5_group3 = pd.read_csv('progress_tstf_nor_3.csv',
                              usecols=['timesteps_total', 'success', 'episode_reward_mean', 'crash', 'out'])


# 为每个算法添加 'algo' 列并标识不同的组
def add_algo_group(df, algo_name):
    df['algo'] = algo_name
    return df


def add_group(df, algo_name):
    df['group'] = algo_name
    return df


# 添加算法和组标识
df_algo1_group1 = add_algo_group(df_algo1_group1, 'IPPO')
df_algo1_group2 = add_algo_group(df_algo1_group2, 'IPPO')
df_algo1_group3 = add_algo_group(df_algo1_group3, 'IPPO')
df_algo1_group1 = add_group(df_algo1_group1, 'IPPO_1')
df_algo1_group2 = add_group(df_algo1_group2, 'IPPO_2')
df_algo1_group3 = add_group(df_algo1_group3, 'IPPO_3')
df_algo2_group1 = add_algo_group(df_algo2_group1, 'CoPO')
df_algo2_group2 = add_algo_group(df_algo2_group2, 'CoPO')
df_algo2_group3 = add_algo_group(df_algo2_group3, 'CoPO')
df_algo2_group1 = add_group(df_algo2_group1, 'CoPO_1')
df_algo2_group2 = add_group(df_algo2_group2, 'CoPO_2')
df_algo2_group3 = add_group(df_algo2_group3, 'CoPO_3')
df_algo3_group1 = add_algo_group(df_algo3_group1, 'CCPPO_Meanfiled')
df_algo3_group2 = add_algo_group(df_algo3_group2, 'CCPPO_Meanfiled')
df_algo3_group3 = add_algo_group(df_algo3_group3, 'CCPPO_Meanfiled')
df_algo3_group1 = add_group(df_algo3_group1, 'CCPPO_Meanfiled_1')
df_algo3_group2 = add_group(df_algo3_group2, 'CCPPO_Meanfiled_2')
df_algo3_group3 = add_group(df_algo3_group3, 'CCPPO_Meanfiled_3')
df_algo4_group1 = add_algo_group(df_algo4_group1, 'CCPPO_Concat')
df_algo4_group2 = add_algo_group(df_algo4_group2, 'CCPPO_Concat')
df_algo4_group3 = add_algo_group(df_algo4_group3, 'CCPPO_Concat')
df_algo4_group1 = add_group(df_algo4_group1, 'CCPPO_Concat_1')
df_algo4_group2 = add_group(df_algo4_group2, 'CCPPO_Concat_2')
df_algo4_group3 = add_group(df_algo4_group3, 'CCPPO_Concat_3')
df_algo5_group1 = add_algo_group(df_algo5_group1, 'TSTF')
df_algo5_group2 = add_algo_group(df_algo5_group2, 'TSTF')
df_algo5_group3 = add_algo_group(df_algo5_group3, 'TSTF')
df_algo5_group1 = add_group(df_algo5_group1, 'TSTF_1')
df_algo5_group2 = add_group(df_algo5_group2, 'TSTF_2')
df_algo5_group3 = add_group(df_algo5_group3, 'TSTF_3')

# 合并所有数据到一个DataFrame中
plot_df = pd.concat([
    df_algo1_group1, df_algo1_group2, df_algo1_group3,
    df_algo2_group1, df_algo2_group2, df_algo2_group3,
    df_algo3_group1, df_algo3_group2, df_algo3_group3,
    df_algo4_group1, df_algo4_group2, df_algo4_group3,
    df_algo5_group1, df_algo5_group2, df_algo5_group3
])
plot_df = plot_df.reset_index(drop=True)

last_100_df = plot_df.groupby(['algo', 'group']).apply(lambda df: df.tail(100)).reset_index(drop=True)

# Calculate the mean for each algorithm and each group over the last 100 data points
stats_df = last_100_df.groupby(['algo']).agg({
    'success': ['mean', 'std'],
    'episode_reward_mean': ['mean', 'std'],
    'crash': ['mean', 'std'],
    'out': ['mean', 'std'],
}).reset_index()
stats_df.columns = ['_'.join(col).strip('_') for col in stats_df.columns.values]
stats_df.to_csv('comparsion.csv', index=False)
print(stats_df)

# 设置绘图的x轴和y轴标签
x = "timesteps_total"
y_labels = ['success', 'episode_reward_mean', 'crash', 'out']
y_titles = ['Success', 'Episode Reward', 'Crash', 'Out']

# 设置颜色和样式
c = sns.color_palette("colorblind", n_colors=5)  # 使用更多颜色来确保每个算法都有不同的颜色
sns.set("talk", "darkgrid")

# 创建一个 2x2 的子图布局，每个子图的高宽比为 1:1.2
fig, axes = plt.subplots(2, 2, figsize=(24, 24))

# for ax in axes.flat:
#     ax.set_aspect(1/1.2)

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 45

# 绘制三个图
algo_colors = {
    'IPPO': c[0],  # 蓝色
    'CoPO': c[1],  # 橙色
    'CCPPO_Meanfiled': c[2],  # 绿色
    'CCPPO_Concat': c[3],  # 红色
    'TSTF': c[4]  # 紫色
}

for i, y in enumerate(y_labels):
    # 使用 seaborn 的 lineplot 绘制图表
    sns.lineplot(x=x, y=y, data=plot_df, hue='algo', palette=algo_colors, ax=axes[i // 2][i % 2], errorbar='sd',
                 alpha=0.7, linewidth=3)

    # 设置每个图的标题和标签
    # axes[i//2][i%2].set_title(f'{y_titles[i]}', fontsize=12)
    axes[i // 2][i % 2].set_xlabel('Time Step', fontsize=40)
    axes[i // 2][i % 2].set_ylabel(y_titles[i], fontsize=40)
    if i < len(y_labels) - 1:
        axes[i // 2][i % 2].legend().set_visible(False)
    axes[i // 2][i % 2].tick_params(axis='both', which='major', labelsize=35)
    axes[i // 2][i % 2].xaxis.get_offset_text().set_fontsize(30)

handles, labels = axes[-1][-1].get_legend_handles_labels()
axes[-1][-1].legend(handles=handles, labels=labels, loc="upper right",
                    bbox_to_anchor=(1, 1), borderaxespad=0.1, frameon=False, fontsize=35)

# 调整布局
plt.tight_layout()

# 设置总标题
# plt.suptitle('Comparison of Algorithms on Multiple Metrics', fontsize=24, y=1.05)

# handles, labels = axes[-1][-1].get_legend_handles_labels()  # 获取第一个子图的图例句柄和标签
# plt.legend(handles=handles, labels=labels,
#            loc="upper right", bbox_to_anchor=(1, 1), frameon=False,  fontsize=35)

plt.xticks(fontsize=40)
# 保存图像
plt.savefig('comparison.pdf', format='pdf', dpi=300, bbox_inches="tight")

# 显示图像
plt.show()
