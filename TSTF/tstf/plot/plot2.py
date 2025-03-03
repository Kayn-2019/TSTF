import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator
# 读取每个算法的三组CSV文件，分别读取timesteps_total和success, episode_reward_mean, crash列
df_algo1_group1 = pd.read_csv('progress_tstf_nor_1.csv',
                              usecols=['timesteps_total', 'success', 'episode_reward_mean', 'crash', 'out'])
df_algo1_group2 = pd.read_csv('progress_tstf_nor_2.csv',
                              usecols=['timesteps_total', 'success', 'episode_reward_mean', 'crash', 'out'])
df_algo1_group3 = pd.read_csv('progress_tstf_nor_3.csv',
                              usecols=['timesteps_total', 'success', 'episode_reward_mean', 'crash', 'out'])

df_algo2_group1 = pd.read_csv('progress_tstf_spa_1.csv',
                              usecols=['timesteps_total', 'success', 'episode_reward_mean', 'crash', 'out'])
df_algo2_group2 = pd.read_csv('progress_tstf_spa_2.csv',
                              usecols=['timesteps_total', 'success', 'episode_reward_mean', 'crash', 'out'])
df_algo2_group3 = pd.read_csv('progress_tstf_spa_3.csv',
                              usecols=['timesteps_total', 'success', 'episode_reward_mean', 'crash', 'out'])

df_algo3_group1 = pd.read_csv('progress_tstf_tem_1.csv',
                              usecols=['timesteps_total', 'success', 'episode_reward_mean', 'crash', 'out'])
df_algo3_group2 = pd.read_csv('progress_tstf_tem_2.csv',
                              usecols=['timesteps_total', 'success', 'episode_reward_mean', 'crash', 'out'])
df_algo3_group3 = pd.read_csv('progress_tstf_tem_3.csv',
                              usecols=['timesteps_total', 'success', 'episode_reward_mean', 'crash', 'out'])

df_algo4_group1 = pd.read_csv('progress_tstf_simsiam_1.csv',
                              usecols=['timesteps_total', 'success', 'episode_reward_mean', 'crash', 'out'])
df_algo4_group2 = pd.read_csv('progress_tstf_simsiam_2.csv',
                              usecols=['timesteps_total', 'success', 'episode_reward_mean', 'crash', 'out'])
df_algo4_group3 = pd.read_csv('progress_tstf_simsiam_3.csv',
                              usecols=['timesteps_total', 'success', 'episode_reward_mean', 'crash', 'out'])


# 为每个算法添加 'algo' 列并标识不同的组
def add_algo_group(df, algo_name):
    df['algo'] = algo_name
    return df

def add_group(df, algo_name):
    df['group'] = algo_name
    return df


# 添加算法和组标识
df_algo1_group1 = add_algo_group(df_algo1_group1, 'TSTF')
df_algo1_group2 = add_algo_group(df_algo1_group2, 'TSTF')
df_algo1_group3 = add_algo_group(df_algo1_group3, 'TSTF')
df_algo1_group1 = add_group(df_algo1_group1, 'TSTF_1')
df_algo1_group2 = add_group(df_algo1_group2, 'TSTF_2')
df_algo1_group3 = add_group(df_algo1_group3, 'TSTF_3')
df_algo2_group1 = add_algo_group(df_algo2_group1, 'TSTF_without_TT')
df_algo2_group2 = add_algo_group(df_algo2_group2, 'TSTF_without_TT')
df_algo2_group3 = add_algo_group(df_algo2_group3, 'TSTF_without_TT')
df_algo2_group1 = add_group(df_algo2_group1, 'TSTF_without_TT_1')
df_algo2_group2 = add_group(df_algo2_group2, 'TSTF_without_TT_2')
df_algo2_group3 = add_group(df_algo2_group3, 'TSTF_without_TT_3')

df_algo3_group1 = add_algo_group(df_algo3_group1, 'TSTF_without_ST')
df_algo3_group2 = add_algo_group(df_algo3_group2, 'TSTF_without_ST')
df_algo3_group3 = add_algo_group(df_algo3_group3, 'TSTF_without_ST')
df_algo3_group1 = add_group(df_algo3_group1, 'TSTF_without_ST_1')
df_algo3_group2 = add_group(df_algo3_group2, 'TSTF_without_ST_2')
df_algo3_group3 = add_group(df_algo3_group3, 'TSTF_without_ST_3')
df_algo4_group1 = add_algo_group(df_algo4_group1, 'TSTF_without_Siam')
df_algo4_group2 = add_algo_group(df_algo4_group2, 'TSTF_without_Siam')
df_algo4_group3 = add_algo_group(df_algo4_group3, 'TSTF_without_Siam')
df_algo4_group1 = add_group(df_algo4_group1, 'TSTF_without_Siam_1')
df_algo4_group2 = add_group(df_algo4_group2, 'TSTF_without_Siam_2')
df_algo4_group3 = add_group(df_algo4_group3, 'TSTF_without_Siam_3')

# 合并所有数据到一个DataFrame中
plot_df = pd.concat([
    df_algo1_group1, df_algo1_group2, df_algo1_group3,
    df_algo4_group1, df_algo4_group2, df_algo4_group3,
    df_algo3_group1, df_algo3_group2, df_algo3_group3,
    df_algo2_group1, df_algo2_group2, df_algo2_group3,


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
desired_order = ['TSTF', 'TSTF_without_Siam', 'TSTF_without_ST', 'TSTF_without_TT']
stats_df = stats_df.set_index('algo').loc[desired_order].reset_index()
stats_df.columns = ['_'.join(col).strip('_') for col in stats_df.columns.values]
stats_df.to_csv('ablation.csv', index=False)
print(stats_df)

# 设置绘图的x轴和y轴标签
x = "timesteps_total"
y_labels = 'success'
y_titles = 'Success'

# 设置颜色和样式
c = sns.color_palette("colorblind", n_colors=4)  # 使用更多颜色来确保每个算法都有不同的颜色
sns.set("talk", "darkgrid")

fig, axes = plt.subplots(1, 2, figsize=(24, 12))
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 20

sns.lineplot(x=x, y=y_labels, data=plot_df, hue='algo', palette=c, ax=axes[0], errorbar='sd',
             linewidth=3)

axes[0].set_xlabel('Time Step', fontsize=40)
axes[0].set_ylabel(y_titles, fontsize=40)
# axes[0].legend().set_visible(False)
# axes[0].legend(loc="lower right", bbox_to_anchor=(1.0, 0.0), borderaxespad=0.1,
#                      frameon=False, fontsize=40,)
axes[0].tick_params(axis='both', which='major', labelsize=35)
axes[0].xaxis.get_offset_text().set_fontsize(30)
# 获取当前图例的handles和labels
handles, labels = axes[0].get_legend_handles_labels()

# 定义映射关系
mapping = {
    'TSTF': 'TSTF',
    'TSTF_without_ST': 'w/o_ST',
    'TSTF_without_TT': 'w/o_TT',
    'TSTF_without_Siam': 'w/o_Siam'
}

# 注意有时第一个label可能是图例标题，可以过滤掉或者直接映射（如果它在mapping中）
new_labels = [mapping.get(label, label) for label in labels]

# 重新设置图例
axes[0].legend(handles=handles, labels=new_labels, loc="lower right",
               bbox_to_anchor=(1.0, 0.0), borderaxespad=0.1, frameon=False, fontsize=35)



axes[1].bar(stats_df['algo'], stats_df['success_mean'],
       yerr=stats_df['success_std'], capsize=10,
       color=c, edgecolor='black')
axes[1].set_xlabel('Ablated Module', fontsize=40)
axes[1].set_ylabel('Success', fontsize=40)
axes[1].tick_params(axis='both', which='major', labelsize=35)
mapping = {
    'TSTF': 'TSTF',
    'TSTF_without_TT': 'w/o_TT',
    'TSTF_without_ST': 'w/o_ST',
    'TSTF_without_Siam': 'w/o_Siam'
}
new_labels = [mapping.get(label, label) for label in stats_df['algo']]
ticks = axes[1].get_xticks()
# 用 FixedLocator 固定这些位置
axes[1].xaxis.set_major_locator(FixedLocator(ticks))
# 然后设置新的 tick labels
axes[1].set_xticklabels(new_labels, fontsize=36)






# 保存图像
plt.savefig('ablation.pdf', format='pdf', dpi=300, bbox_inches="tight")

# 显示图像
plt.show()
