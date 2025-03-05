import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

title_font =   {'family':'STIXGeneral',
    'style':'normal',
    'weight':'bold',
    'size': 12}

lable_font =   {'family':'STIXGeneral',
    'style':'normal',
    'weight':'bold',
    'size': 10}

axis_font =   {'family':'STIXGeneral',
    'style':'normal',
    'weight':'bold',
    'size': 10}

legend_font =   {'family':'STIXGeneral',
    'style':'normal',
    'weight':'bold',
    'size': 8}

# 假设文件路径分别为 'method1.csv' 和 'method2.csv'
def group_selection(df, group_id):
    conditions = [
        df[group_id].between(1, 2),
        df[group_id].between(3, 4),
        df[group_id].between(5, 6),
        df[group_id] > 6,
    ]
    groups = ['1-2', '3-4', '5-6', '7+']
    
    df.loc[:, 'group'] = np.select(conditions, groups, default='Unknown')

    return df

def group_metrics(df, metric):
    group_metrics = df.groupby('group').apply(
        lambda x: pd.Series({
            'group_metric': (x[metric] * x['sample size']).sum() / x['sample size'].sum(),
            'group_size': x['sample size'].sum()
        })
    )
    return group_metrics

    # 计算每个分组的加权指标
if __name__ == "__main__":
    dataset_1 = 'Receipt'
    dataset_2 = 'helpdesk2017'
    baseline_path = '/home/inspur/zhengchao/baselines/PPM'
    file_path_1_lstm = '{}/ProcessSequencePrediction/code/output_files/time-spilt/results/{}/next_activity.csv'.format(baseline_path, dataset_1)
    file_path_1_sufftrans = '{}/MiDA/Results-time/{}/accuracy/next_activity.csv'.format(baseline_path, dataset_1)

    file_path_2_lstm = '{}/ProcessSequencePrediction/code/output_files/time-spilt/results/{}/next_activity.csv'.format(baseline_path, dataset_2)
    file_path_2_sufftrans = '{}/MiDA/Results-time/{}/accuracy/next_activity.csv'.format(baseline_path, dataset_2)

    # 读取数据
    df_1 = pd.read_csv(file_path_1_lstm)
    df_2 = pd.read_csv(file_path_1_sufftrans)
    df_3 = pd.read_csv(file_path_2_lstm)
    df_4 = pd.read_csv(file_path_2_sufftrans)


    # 排除最后一行（整体性能）
    df_grouped_1 = df_1[1:-1]
    df_grouped_2 = df_2[1:-1]
    df_grouped_3 = df_3[1:-1]
    df_grouped_4 = df_4[1:-1]
    

    df_grouped_1 = group_selection(df_grouped_1, 'suffix_var_num')
    df_grouped_2 = group_selection(df_grouped_2, 'suffix_var_num')
    df_grouped_3 = group_selection(df_grouped_3,'suffix_var_num')
    df_grouped_4 = group_selection(df_grouped_4,'suffix_var_num')
    
    group_metrics_1 = group_metrics(df_grouped_1, 'accuracy')
    group_metrics_2 = group_metrics(df_grouped_2, 'accuracy')
    group_metrics_3 = group_metrics(df_grouped_3, 'accuracy')
    group_metrics_4 = group_metrics(df_grouped_4, 'accuracy')
    
    PTC_variant_sample_size = [6074,1661,392,2148]
    Helpdesk_variant_sample_size = [1284,7602,334,2528]

    # 设置柱状图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4))  # 创建两个子图
    width = 0.4  # 每个柱子的宽度
    # 绘制每个分组的平均准确率，分别对应两个方法
    group_labels = ['1-2', '3-4', '5-6', '7+']
    x = np.arange(len(group_labels))

    ax1.bar(x - width/2, group_metrics_1['group_metric'], width, label='LSTM', color='skyblue')
    ax1.bar(x + width/2, group_metrics_2['group_metric'], width, label='MiDA', color='orange')
    ax2.bar(x - width/2, group_metrics_3['group_metric'], width, label='LSTM', color='skyblue')
    ax2.bar(x + width/2, group_metrics_4['group_metric'], width, label='MiDA', color='orange')

    # 在第一个子图上绘制BPIC2020样本数线条
    ax1_sample = ax1.twinx()  # 创建双 Y 轴
    ax1_sample.plot(x, PTC_variant_sample_size, label='Sample Size', color='green', marker='o', linestyle='-', linewidth=2)
    ax1_sample.set_yticks([])

    # 在第二个子图上绘制Helpdesk样本数线条
    ax2_sample = ax2.twinx()  # 创建双 Y 轴
    ax2_sample.plot(x, Helpdesk_variant_sample_size, label='Sample Size', color='purple', marker='o', linestyle='-', linewidth=2)
    ax2_sample.set_ylabel('Sample Size', fontproperties=lable_font)
    ax2_sample.set_yticks([0, 2000, 4000, 6000, 8000])
    ax2_sample.set_yticklabels([0, 2000, 4000, 6000, 8000], fontproperties=axis_font)

    ax1_sample.set_ylim(bottom=0, top=8000)
    ax2_sample.set_ylim(bottom=0, top=8000)

    ax1_sample.legend(prop=legend_font, frameon=False)
    ax2_sample.legend(prop=legend_font, frameon=False)

    ax1.set_ylim(bottom=0.3, top=1.0)
    ax2.set_ylim(bottom=0.3, top=1.0)

    ax1.set_xticks(x)
    ax1.set_xticklabels(group_labels, fontproperties=axis_font)
    ax1.set_yticks([0.4, 0.6, 0.8, 1.0])
    ax1.set_yticklabels([0.4, 0.6, 0.8, 1.0], fontproperties=axis_font)

    ax2.set_xticks(x)
    ax2.set_xticklabels(group_labels, fontproperties=axis_font)
    ax2.set_yticks([0.4, 0.6, 0.8, 1.0])
    ax2.set_yticklabels([0.4, 0.6, 0.8, 1.0], fontproperties=axis_font)

    # 设置图表标题和标签
    ax1.set_ylabel('Accuracy', fontproperties=lable_font)
    ax1.set_title('BPIC2020_PrepaidTravelCost', fontproperties=title_font)
    ax2.set_title('Helpdesk', fontproperties=title_font)
    ax1.set_xlabel('Suffix Variant Number Group', fontproperties=lable_font)
    ax2.set_xlabel('Suffix Variant Number Group', fontproperties=lable_font)
    plt.xticks(rotation=0)

    # 显示图例
    # 修改图例位置
    ax1.legend(prop=legend_font, frameon=False, loc='upper left', bbox_to_anchor=(0.71, 1))
    ax1_sample.legend(prop=legend_font, frameon=False, loc='upper left', bbox_to_anchor=(0.71, 0.9))
    ax2.legend(prop=legend_font, frameon=False, loc='upper left', bbox_to_anchor=(0.71, 1))
    ax2_sample.legend(prop=legend_font, frameon=False, loc='upper left', bbox_to_anchor=(0.71, 0.9))


    # 显示图表
    plt.tight_layout()
    plt.show()
    plt.savefig('figs/fig1.pdf', format='pdf', transparent=False)
    plt.savefig('figs/fig1.jpg', format='pdf', transparent=False)
