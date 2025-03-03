import json
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
import numpy as np



def print_single():
    # 读入奖励数据
    with open('./sac/582.json', 'r', encoding='utf-8') as fp:
        single = json.load(fp)

    single = single[:500]

    single = list(savgol_filter(single, 41, 3))
    t = list(range(len(single)))

    # 保存处理后的数据
    with open('./sac/582_1.json', 'w') as file_obj:
        json.dump(single, file_obj)

    # 绘制图像
    # plt.style.use('seaborn-v0_8-darkgrid')
    plt.style.use('seaborn-darkgrid')
    fig, ax = plt.subplots()
    plt.plot(t, single, 'blue', label='SAC')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Average Return')
    ax.legend()
    plt.show()


def calculate(data1, data2, data3):
    data = [data1, data2, data3]  # 3*n
    mean = np.mean(data, axis=0)  # 1*100
    std_deviation = np.std(data, axis=0)  # 1*100

    return mean, std_deviation


def print_compare():


    with open('./droq/708_1.json', 'r', encoding='utf-8') as fp:
        droq_single_3 = json.load(fp)
    with open('./droq/717_1.json', 'r', encoding='utf-8') as fp:
        droq_single_2 = json.load(fp)
    with open('./droq/962_1.json', 'r', encoding='utf-8') as fp:
        droq_single_1 = json.load(fp)

    with open('./iqsac/511_1.json', 'r', encoding='utf-8') as fp:
        iqsac_single_3 = json.load(fp)
    with open('./iqsac/574_1.json', 'r', encoding='utf-8') as fp:
        iqsac_single_2 = json.load(fp)
    with open('./iqsac/727_1.json', 'r', encoding='utf-8') as fp:
        iqsac_single_1 = json.load(fp)

    with open('./qrsac/511_1.json', 'r', encoding='utf-8') as fp:
        qrsac_single_1 = json.load(fp)
    with open('./qrsac/qr_sac_2.json', 'r', encoding='utf-8') as fp:
        qrsac_single_2 = json.load(fp)
    with open('./qrsac/602_1.json', 'r', encoding='utf-8') as fp:
        qrsac_single_3 = json.load(fp)

    with open('./sac/515_1.json', 'r', encoding='utf-8') as fp:
        sac_single_1 = json.load(fp)
    with open('./sac/sac_2.json', 'r', encoding='utf-8') as fp:
        sac_single_2 = json.load(fp)
    with open('./sac/582_1.json', 'r', encoding='utf-8') as fp:
        sac_single_3 = json.load(fp)

    with open('./spl/501_1.json', 'r', encoding='utf-8') as fp:
        spl_single_3 = json.load(fp)
    with open('./spl/617_1.json', 'r', encoding='utf-8') as fp:
        spl_single_2 = json.load(fp)
    with open('./spl/775_1.json', 'r', encoding='utf-8') as fp:
        spl_single_1 = json.load(fp)


    x = list(range(len(droq_single_3)))

    droq_mean_curve, droq_std_deviation = calculate(droq_single_1, droq_single_2, droq_single_3)
    iqsac_mean_curve, iqsac_std_deviation = calculate(iqsac_single_1, iqsac_single_2, iqsac_single_3)
    qrsac_mean_curve, qrsac_std_deviation = calculate(qrsac_single_1, qrsac_single_2, qrsac_single_3)
    sac_mean_curve, sac_std_deviation = calculate(sac_single_1, sac_single_2, sac_single_3)
    spl_mean_curve, spl_std_deviation = calculate(spl_single_3, spl_single_2, spl_single_1)

    # 创建图形和坐标轴
    plt.style.use('seaborn-darkgrid')
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']

    fig, ax = plt.subplots()

    # plt.rcParams['font.family'] = 'Times New Roman'

    ax.plot(x, droq_mean_curve, label='ESPL', color='red')
    ax.fill_between(x, droq_mean_curve - droq_std_deviation, droq_mean_curve + droq_std_deviation, alpha=0.2, color='red')

    ax.plot(x, iqsac_mean_curve, label='IQSAC', color='purple')
    ax.fill_between(x, iqsac_mean_curve - iqsac_std_deviation, iqsac_mean_curve + iqsac_std_deviation, alpha=0.2, color='purple')

    ax.plot(x, qrsac_mean_curve, label='QRSAC', color='brown')
    ax.fill_between(x, qrsac_mean_curve - qrsac_std_deviation, qrsac_mean_curve + qrsac_std_deviation, alpha=0.2, color='brown')

    ax.plot(x, sac_mean_curve, label='SAC', color='green')
    ax.fill_between(x, sac_mean_curve - sac_std_deviation, sac_mean_curve + sac_std_deviation, alpha=0.2, color='green')

    ax.plot(x, spl_mean_curve, label='SPL', color='blue')
    ax.fill_between(x, spl_mean_curve - spl_std_deviation, spl_mean_curve + spl_std_deviation, alpha=0.2, color='blue')

    # 添加标签和图例
    ax.tick_params(axis='x', labelsize=15)
    ax.tick_params(axis='y', labelsize=15)
    ax.set_xlabel('Training episode', fontsize='15')
    ax.set_ylabel('Average reward', fontsize='15')
    # ax.legend(loc='lower right', fontsize='13')

    plt.title('Town 01')
    plt.tight_layout()
    plt.savefig('compare_1.svg', format='svg')
    plt.show()

    # 显示图形
    plt.show()


if __name__ == '__main__':
    # print_single()
    print_compare()
