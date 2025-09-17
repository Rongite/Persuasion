import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import argparse
import os
import random

parser = argparse.ArgumentParser(description='Adjust the path and others.')
parser.add_argument('--book', type=str, help='select the book')
parser.add_argument('--technique_dir', type=str, help='create the technique dir')
parser.add_argument('--technique', type=str, help='select the technique')
# parser.add_argument('--original_rouge_score', nargs='+', type=float, help='List of numbers')
args = parser.parse_args()

'''
The first 4 experiment: data processing
'''
# 1. 将四种数据分别集中起来得到四个dataframe
def merge_csv_files(directory, keyword):
    # 初始化一个空的 DataFrame
    combined_df = pd.DataFrame()
    
    # 遍历目录中的所有文件
    for filename in os.listdir(directory):
        # 检查文件名是否包含特定字段并且是 CSV 文件
        if keyword in filename and filename.endswith('.csv'):
            # 构建文件的完整路径
            file_path = os.path.join(directory, filename)
            # 读取 CSV 文件
            df = pd.read_csv(file_path)
            # 将读取的数据追加到合并的 DataFrame 中
            combined_df = pd.concat([combined_df, df], ignore_index=True)
    
    return combined_df

eval_data_dir = f"/home/jlong1/Downloads/persuasion/Data_n_Code_persuasion/jikailoong/3_evaluation_results/{args.book}/{args.technique_dir}"
zero_shot_with_judge = "zero_shot_with_judge"
zero_shot_with_no_judge = "zero_shot_with_no_judge"
few_shots_with_judge = "few_shots_with_judge"
few_shots_with_no_judge = "few_shots_with_no_judge"

zero_shot_with_judge_df = merge_csv_files(eval_data_dir, zero_shot_with_judge)
zero_shot_with_no_judge_df = merge_csv_files(eval_data_dir, zero_shot_with_no_judge)
few_shots_with_judge_df = merge_csv_files(eval_data_dir, few_shots_with_judge)
few_shots_with_no_judge_df = merge_csv_files(eval_data_dir, few_shots_with_no_judge)

# 2. 将这四个dataframe按照模型类别各自分成两个dataframe，目前有4×2个dataframe
def split_dataframe_by_model(df):
    # 假设 'model' 列的值在奇数行和偶数行中分别一致
    # 获取第一个模型的名称（位于第一个数据行，即索引 0）
    model1 = df.iloc[0]['model']
    
    # 创建两个空的 DataFrame 用于存储分割后的数据
    df_model1 = pd.DataFrame(columns=df.columns)
    df_model2 = pd.DataFrame(columns=df.columns)
    
    # 遍历 DataFrame 中的行
    for index, row in df.iterrows():
        # 根据行索引的奇偶性分配到不同的 DataFrame
        if index % 2 == 0:
            df_model1 = pd.concat([df_model1, pd.DataFrame([row])], ignore_index=True)
        else:
            df_model2 = pd.concat([df_model2, pd.DataFrame([row])], ignore_index=True)

    return df_model1, df_model2

zero_shot_with_judge_df_for_claude, zero_shot_with_judge_df_for_gpt = split_dataframe_by_model(zero_shot_with_judge_df)
zero_shot_with_no_judge_df_for_claude, zero_shot_with_no_judge_df_for_gpt = split_dataframe_by_model(zero_shot_with_no_judge_df)
few_shots_with_judge_df_for_claude, few_shots_with_judge_df_for_gpt = split_dataframe_by_model(few_shots_with_judge_df)
few_shots_with_no_judge_df_for_claude, few_shots_with_no_judge_df_for_gpt = split_dataframe_by_model(few_shots_with_no_judge_df)

# 3. 将每种数据对应的两个dataframe按相同的索引分别随机抽取出60，50，40，30，20，10个数据,得到4×2×6个dataframe（8个df list）

def sample_same_rows_sorted_index(df1, df2, num_rows):
    # 确保两个DataFrame行数相同
    assert len(df1) == len(df2), "DataFrames should have the same number of rows"
    
    # 生成随机样本的索引，并对索引进行排序
    sample_indices = df1.sample(n=num_rows, random_state=42).index.sort_values()
    
    # 根据排序后的索引从两个DataFrame中获取行，并重置索引
    sampled_df1 = df1.loc[sample_indices].reset_index(drop=True)
    sampled_df2 = df2.loc[sample_indices].reset_index(drop=True)
    
    return sampled_df1, sampled_df2


def sample_and_add_to_list(df1, df2, group_list):
    df_list_1, df_list_2 = [], []
    for group in group_list:
        sampled_df1, sampled_df2 = sample_same_rows_sorted_index(df1, df2, group)
        df_list_1.append(sampled_df1)
        df_list_2.append(sampled_df2)
    
    return df_list_1, df_list_2


the_group_list = [60, 50, 40, 30, 20, 10]
zero_shot_with_judge_df_list_for_claude, zero_shot_with_judge_df_list_for_gpt = sample_and_add_to_list(zero_shot_with_judge_df_for_claude, zero_shot_with_judge_df_for_gpt, the_group_list)
zero_shot_with_no_judge_df_list_for_claude, zero_shot_with_no_judge_df_list_for_gpt = sample_and_add_to_list(zero_shot_with_no_judge_df_for_claude, zero_shot_with_no_judge_df_for_gpt, the_group_list)
few_shots_with_judge_df_list_for_claude, few_shots_with_judge_df_list_for_gpt = sample_and_add_to_list(few_shots_with_judge_df_for_claude, few_shots_with_judge_df_for_gpt, the_group_list)
few_shots_with_no_judge_df_list_for_claude, few_shots_with_no_judge_df_list_for_gpt = sample_and_add_to_list(few_shots_with_no_judge_df_for_claude, few_shots_with_no_judge_df_for_gpt, the_group_list)

'''
The first 4 experiment: data statistics
'''
# # 1. 将每个dataframe list输入函数，得到每个dataframe的平均值，最大值，最小值，中位数，上四分卫数，下四分位数，以及箱线图（箱线图要体现平均数，和original rouge score）
# def save_table_as_image(df, filename, title, column_name):
#     """
#     Save a DataFrame as an image file with clearly labeled headers for statistics, including a title and column name.
#     """
#     fig, ax = plt.subplots(figsize=(12, 2 + 0.5 * len(df)))  # Adjust size as needed
#     ax.axis('tight')
#     ax.axis('off')
#     table = ax.table(cellText=df.values, colLabels=df.columns, cellLoc='center', loc='center', colColours=['#f5f5f5']*df.shape[1])
#     table.auto_set_font_size(False)
#     table.set_fontsize(10)
#     table.scale(1.2, 1.2)
#     plt.title(f"{title} - {column_name}", fontsize=14, fontweight='bold', pad=20)
#     plt.savefig(filename, dpi=200, bbox_inches='tight')
#     plt.close()

# def generate_statistics_and_plots(df_list, names_list, cols, output_dir, titles):
#     """
#     Generates and saves statistical summaries and boxplots for specified columns from a list of DataFrames.
#     Includes titles and column names.
#     """
#     if not os.path.exists(output_dir):
#         os.makedirs(output_dir)

#     for col, title in zip(cols, titles):
#         combined_stats = []
#         for df, name in zip(df_list, names_list):
#             stats = df[col].describe(percentiles=[0.25, 0.5, 0.75]).transpose()
#             stats = pd.DataFrame(stats).transpose()
#             stats = stats.round(6)  # Round to six decimal places
#             stats['sample size'] = name
#             combined_stats.append(stats[['sample size', 'mean', 'min', '25%', '50%', '75%', 'max']])
        
#         summary_df = pd.concat(combined_stats)
#         summary_df.columns = ['sample size', 'Mean', 'Min', '25th Percentile', 'Median', '75th Percentile', 'Max']
#         summary_table_filename = os.path.join(output_dir, f"{title}_summary_{col}.png")
#         save_table_as_image(summary_df, summary_table_filename, title, col)
#         print(f"Saved summary table to {summary_table_filename}")

#         # Save to CSV as well
#         csv_filename = summary_table_filename.replace('.png', '.csv')
#         summary_df.to_csv(csv_filename, index=False)
#         print(f"Saved summary table to CSV {csv_filename}")

#         plt.figure(figsize=(12, 6))
#         data = [df[col].dropna() for df in df_list]
#         plt.boxplot(data, labels=names_list)
#         plt.title(f'Distribution of {col} - {title}', fontsize=14)
#         plt.ylabel('Values')
#         plt.xlabel('sample sizes')
#         plt.grid(True)
#         plt.xticks(rotation=45)
#         plot_filename = os.path.join(output_dir, f"{title}_{col}.png")
#         plt.savefig(plot_filename)
#         plt.close()
#         print(f"Saved plot to {plot_filename}")

def save_table_as_image(df, filename, title, column_name):
    """
    Save a DataFrame as a table in a JPG file with large fonts.
    """
    fig, ax = plt.subplots(figsize=(16, 4 + 0.7 * len(df)))
    ax.axis('tight')
    ax.axis('off')

    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        cellLoc='center',
        loc='center',
        colColours=['#f5f5f5'] * df.shape[1]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(18)
    table.scale(1.8, 1.8)

    fig.text(0.5, 0.98, f"{title} - {column_name}", fontsize=22, fontweight='bold', ha='center')

    fig.savefig(filename, format='jpg', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved summary table to {filename}")


def generate_statistics_and_plots(df_list, names_list, cols, output_dir, titles, original_scores=None):
    """
    Generates and saves statistical summaries and boxplots for specified columns from a list of DataFrames.
    Saves outputs as JPGs with large fonts.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for col, title in zip(cols, titles):
        model_name = title.split(" - ")[0]
        if col == "rouge1_precision":
            title_name = "ROUGE-1"
        elif col == "rougeL_precision":
            title_name = "ROUGE-L"
        else:
            title_name = col

        combined_stats = []
        for df, name in zip(df_list, names_list):
            stats = df[col].describe(percentiles=[0.25, 0.5, 0.75]).transpose()
            stats = pd.DataFrame(stats).transpose()
            stats = stats.round(6)
            stats['Sample Size'] = name
            combined_stats.append(stats[['Sample Size', 'mean', 'min', '25%', '50%', '75%', 'max']])

        summary_df = pd.concat(combined_stats)
        summary_df.columns = ['Sample Size', 'Mean', 'Min', '25th Percentile', 'Median', '75th Percentile', 'Max']

        summary_table_filename = os.path.join(output_dir, f"{title}_summary_{col}.jpg")
        save_table_as_image(summary_df, summary_table_filename, title, col)

        csv_filename = summary_table_filename.replace('.jpg', '.csv')
        summary_df.to_csv(csv_filename, index=False)
        print(f"Saved summary table to CSV {csv_filename}")

        # Save boxplot
        jpg_plot_filename = os.path.join(output_dir, f"{title}_{col}.jpg")
        fig, ax = plt.subplots(figsize=(16, 10))
        data = [df[col].dropna() for df in df_list]
        box = ax.boxplot(data, labels=names_list, patch_artist=True)

        for box_part in box['boxes']:
            box_part.set(facecolor="white", edgecolor="black", linewidth=2)

        for median in box['medians']:
            median.set(color='orangered', linewidth=2)

        full_title = f"{title_name} - {model_name}"
        ax.set_title(full_title, fontsize=20, fontweight='bold', loc='center', color='black')

        ax.set_ylabel('Values', fontsize=16)
        ax.set_xlabel('Sample Size', fontsize=16)
        ax.grid(True)
        plt.xticks(fontsize=14, rotation=45)
        plt.yticks(fontsize=14)

        fig.savefig(jpg_plot_filename, format='jpg', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved plot to {jpg_plot_filename}")


sample_size_list = ["60", "50", "40", "30", "20", "10"]
cols = ['rouge1_precision', 'rougeL_precision']

zero_shot_with_judge_output_dir = f"/home/jlong1/Downloads/persuasion/Data_n_Code_persuasion/jikailoong/3_evaluation_results/{args.book}/{args.technique_dir}/statistical_report/zero_shot_with_judge"
if not os.path.exists(zero_shot_with_judge_output_dir):
    # If it does not exist, create it
    os.makedirs(zero_shot_with_judge_output_dir)
titles_1_1 = ['Claude-3-haiku - zero_shot_with_judge', 'Claude-3-haiku - zero_shot_with_judge']
generate_statistics_and_plots(zero_shot_with_judge_df_list_for_claude, sample_size_list, cols, zero_shot_with_judge_output_dir, titles_1_1)
titles_1_2 = ['GPT-4o-mini - zero_shot_with_judge', 'GPT-4o-mini - zero_shot_with_judge']
generate_statistics_and_plots(zero_shot_with_judge_df_list_for_gpt, sample_size_list, cols, zero_shot_with_judge_output_dir, titles_1_2)

zero_shot_with_no_judge_output_dir = f"/home/jlong1/Downloads/persuasion/Data_n_Code_persuasion/jikailoong/3_evaluation_results/{args.book}/{args.technique_dir}/statistical_report/zero_shot_with_no_judge"
if not os.path.exists(zero_shot_with_no_judge_output_dir):
    # If it does not exist, create it
    os.makedirs(zero_shot_with_no_judge_output_dir)
titles_2_1 = ['Claude-3-haiku - zero_shot_with_no_judge', 'Claude-3-haiku - zero_shot_with_no_judge']
generate_statistics_and_plots(zero_shot_with_no_judge_df_list_for_claude, sample_size_list, cols, zero_shot_with_no_judge_output_dir, titles_2_1)
titles_2_2 = ['GPT-4o-mini - zero_shot_with_no_judge', 'GPT-4o-mini - zero_shot_with_no_judge']
generate_statistics_and_plots(zero_shot_with_no_judge_df_list_for_gpt, sample_size_list, cols, zero_shot_with_no_judge_output_dir, titles_2_2)

few_shots_with_judge_output_dir = f"/home/jlong1/Downloads/persuasion/Data_n_Code_persuasion/jikailoong/3_evaluation_results/{args.book}/{args.technique_dir}/statistical_report/few_shots_with_judge"
if not os.path.exists(few_shots_with_judge_output_dir):
    # If it does not exist, create it
    os.makedirs(few_shots_with_judge_output_dir)
titles_3_1 = ['Claude-3-haiku - few_shots_with_judge', 'Claude-3-haiku - few_shots_with_judge']
generate_statistics_and_plots(few_shots_with_judge_df_list_for_claude, sample_size_list, cols, few_shots_with_judge_output_dir, titles_3_1)
titles_3_2 = ['GPT-4o-mini - few_shots_with_judge', 'GPT-4o-mini - few_shots_with_judge']
generate_statistics_and_plots(few_shots_with_judge_df_list_for_gpt, sample_size_list, cols, few_shots_with_judge_output_dir, titles_3_2)

few_shots_with_no_judge_output_dir = f"/home/jlong1/Downloads/persuasion/Data_n_Code_persuasion/jikailoong/3_evaluation_results/{args.book}/{args.technique_dir}/statistical_report/few_shots_with_no_judge"
if not os.path.exists(few_shots_with_no_judge_output_dir):
    # If it does not exist, create it
    os.makedirs(few_shots_with_no_judge_output_dir)
titles_4_1 = ['Claude-3-haiku - few_shots_with_no_judge', 'Claude-3-haiku - few_shots_with_no_judge']
generate_statistics_and_plots(few_shots_with_no_judge_df_list_for_claude, sample_size_list, cols, few_shots_with_no_judge_output_dir, titles_4_1)
titles_4_2 = ['GPT-4o-mini - few_shots_with_no_judge', 'GPT-4o-mini - few_shots_with_no_judge']
generate_statistics_and_plots(few_shots_with_no_judge_df_list_for_gpt, sample_size_list, cols, few_shots_with_no_judge_output_dir, titles_4_2)

'''
The 5th experiment: data processing
'''
# 1. 将四种数据的各自从20个inference_scaling_data文件随机挑出20，15，10，5个文件，画出4×4个dataframe
def merge_random_csv(directory, num_files):
    """
    随机选择并合并指定数量的CSV文件。

    参数:
    directory (str): 包含CSV文件的目录路径。
    num_files (int): 需要合并的CSV文件数量。

    返回:
    pandas.DataFrame: 合并后的DataFrame。
    """
    # 设定随机种子，确保每次选择的文件相同
    random.seed(42)  

    # 获取目录中所有的CSV文件
    files = [f for f in os.listdir(directory) if f.endswith('.csv')]
    
    # 随机选择特定数量的CSV文件
    if len(files) < num_files:
        print("Warning: Requested more files than are available. Using all available files.")
        selected_files = files
    else:
        selected_files = random.sample(files, num_files)

    # 读取并合并CSV文件
    df_list = [pd.read_csv(os.path.join(directory, file)) for file in selected_files]
    combined_df = pd.concat(df_list, ignore_index=True)

    return combined_df

# zero_shot_with_judge inference scaling 20、15、10、5次
zero_shot_with_judge_inference_scaling_dir = f"/home/jlong1/Downloads/persuasion/Data_n_Code_persuasion/jikailoong/3_evaluation_results/{args.book}/{args.technique_dir}/1_zero_shot_with_judge"
zero_shot_with_judge_scaling_20_df = merge_random_csv(zero_shot_with_judge_inference_scaling_dir, 20)
zero_shot_with_judge_scaling_15_df = merge_random_csv(zero_shot_with_judge_inference_scaling_dir, 15)
zero_shot_with_judge_scaling_10_df = merge_random_csv(zero_shot_with_judge_inference_scaling_dir, 10)
zero_shot_with_judge_scaling_5_df = merge_random_csv(zero_shot_with_judge_inference_scaling_dir, 5)

# zero_shot_without_judge inference scaling 20、15、10、5次
zero_shot_without_judge_inference_scaling_dir = f"/home/jlong1/Downloads/persuasion/Data_n_Code_persuasion/jikailoong/3_evaluation_results/{args.book}/{args.technique_dir}/2_zero_shot_without_judge"
zero_shot_without_judge_scaling_20_df = merge_random_csv(zero_shot_without_judge_inference_scaling_dir, 20)
zero_shot_without_judge_scaling_15_df = merge_random_csv(zero_shot_without_judge_inference_scaling_dir, 15)
zero_shot_without_judge_scaling_10_df = merge_random_csv(zero_shot_without_judge_inference_scaling_dir, 10)
zero_shot_without_judge_scaling_5_df = merge_random_csv(zero_shot_without_judge_inference_scaling_dir, 5)

# few_shots_with_judge inference scaling 20、15、10、5次
few_shots_with_judge_inference_scaling_dir = f"/home/jlong1/Downloads/persuasion/Data_n_Code_persuasion/jikailoong/3_evaluation_results/{args.book}/{args.technique_dir}/3_few_shots_with_judge"
few_shots_with_judge_scaling_20_df = merge_random_csv(few_shots_with_judge_inference_scaling_dir, 20)
few_shots_with_judge_scaling_15_df = merge_random_csv(few_shots_with_judge_inference_scaling_dir, 15)
few_shots_with_judge_scaling_10_df = merge_random_csv(few_shots_with_judge_inference_scaling_dir, 10)
few_shots_with_judge_scaling_5_df = merge_random_csv(few_shots_with_judge_inference_scaling_dir, 5)

# few_shots_without_judge inference scaling 20、15、10、5次
few_shots_without_judge_inference_scaling_dir = f"/home/jlong1/Downloads/persuasion/Data_n_Code_persuasion/jikailoong/3_evaluation_results/{args.book}/{args.technique_dir}/4_few_shots_without_judge"
few_shots_without_judge_scaling_20_df = merge_random_csv(few_shots_without_judge_inference_scaling_dir, 20)
few_shots_without_judge_scaling_15_df = merge_random_csv(few_shots_without_judge_inference_scaling_dir, 15)
few_shots_without_judge_scaling_10_df = merge_random_csv(few_shots_without_judge_inference_scaling_dir, 10)
few_shots_without_judge_scaling_5_df = merge_random_csv(few_shots_without_judge_inference_scaling_dir, 5)

# 2. 将这十六个dataframe按照模型类别各自分成两个dataframe，目前有4×4×2个dataframe（8个df list）
zero_shot_with_judge_scaling_list = [zero_shot_with_judge_scaling_20_df, zero_shot_with_judge_scaling_15_df, zero_shot_with_judge_scaling_10_df, zero_shot_with_judge_scaling_5_df]
zero_shot_without_judge_scaling_list = [zero_shot_without_judge_scaling_20_df, zero_shot_without_judge_scaling_15_df, zero_shot_without_judge_scaling_10_df, zero_shot_without_judge_scaling_5_df]
few_shots_with_judge_scaling_list = [few_shots_with_judge_scaling_20_df, few_shots_with_judge_scaling_15_df, few_shots_with_judge_scaling_10_df, few_shots_with_judge_scaling_5_df]
few_shots_without_judge_scaling_list = [few_shots_without_judge_scaling_20_df, few_shots_without_judge_scaling_15_df, few_shots_without_judge_scaling_10_df, few_shots_without_judge_scaling_5_df]

def split_the_df_list(df_list):
    claude_df_list = []
    gpt_df_list = []
    for df in df_list:
        df1, df2 = split_dataframe_by_model(df)
        claude_df_list.append(df1)
        gpt_df_list.append(df2)

    return claude_df_list, gpt_df_list

zero_shot_with_judge_scaling_list_for_claude, zero_shot_with_judge_scaling_list_for_gpt = split_the_df_list(zero_shot_with_judge_scaling_list)
zero_shot_without_judge_scaling_list_for_claude, zero_shot_without_judge_scaling_list_for_gpt = split_the_df_list(zero_shot_without_judge_scaling_list)
few_shots_with_judge_scaling_list_for_claude, few_shots_with_judge_scaling_list_for_gpt = split_the_df_list(few_shots_with_judge_scaling_list)
few_shots_without_judge_scaling_list_for_claude, few_shots_without_judge_scaling_list_for_gpt = split_the_df_list(few_shots_without_judge_scaling_list)
'''
The 5th experiment: data statistics
'''
def generate_statistics_and_plots_2(df_list, names_list, cols, output_dir, titles, original_scores=None):
    """
    Generates and saves statistical summaries and boxplots for specified columns from a list of DataFrames.
    Saves outputs as PDFs with large fonts.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for col, title in zip(cols, titles):
        model_name = title.split(" - ")[0]
        if col == "rouge1_precision":
            title_name = "ROUGE-1"
        elif col == "rougeL_precision":
            title_name = "ROUGE-L"
        else:
            title_name = col

        combined_stats = []
        for df, name in zip(df_list, names_list):
            stats = df[col].describe(percentiles=[0.25, 0.5, 0.75]).transpose()
            stats = pd.DataFrame(stats).transpose()
            stats = stats.round(6)
            stats['Sample Size'] = name
            combined_stats.append(stats[['Sample Size', 'mean', 'min', '25%', '50%', '75%', 'max']])

        summary_df = pd.concat(combined_stats)
        summary_df.columns = ['Sample Size', 'Mean', 'Min', '25th Percentile', 'Median', '75th Percentile', 'Max']

        summary_table_filename = os.path.join(output_dir, f"{title}_summary_{col}.jpg")
        save_table_as_image(summary_df, summary_table_filename, title, col)

        csv_filename = summary_table_filename.replace('.jpg', '.csv')
        summary_df.to_csv(csv_filename, index=False)
        print(f"Saved summary table to CSV {csv_filename}")

        # Save boxplot
        jpg_plot_filename = os.path.join(output_dir, f"{title}_{col}.jpg")
        fig, ax = plt.subplots(figsize=(16, 10))
        data = [df[col].dropna() for df in df_list]
        box = ax.boxplot(data, labels=names_list, patch_artist=True)

        for box_part in box['boxes']:
            box_part.set(facecolor="white", edgecolor="black", linewidth=2)

        for median in box['medians']:
            median.set(color='orangered', linewidth=2)

        full_title = f"{title_name} - {model_name}"
        ax.set_title(full_title, fontsize=20, fontweight='bold', loc='center', color='black')

        ax.set_ylabel('Values', fontsize=16)
        ax.set_xlabel('Sample Size', fontsize=16)
        ax.grid(True)
        plt.xticks(fontsize=14, rotation=45)
        plt.yticks(fontsize=14)

        fig.savefig(jpg_plot_filename, format='jpg', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved plot to {jpg_plot_filename}")

# 1. 将每个dataframe输入函数，得到每个dataframe的平均值，最大值，最小值，中位数，上四分卫数，下四分位数，以及箱线图（箱线图要体现平均数，和original rouge score）
sample_size_list_2 = ["20", "15", "10", "5"]
cols = ['rouge1_precision', 'rougeL_precision']

zero_shot_with_judge_scaling_output_dir = f"/home/jlong1/Downloads/persuasion/Data_n_Code_persuasion/jikailoong/3_evaluation_results/{args.book}/{args.technique_dir}/statistical_report/inference_scaling/zero_shot_with_judge"
if not os.path.exists(zero_shot_with_judge_scaling_output_dir):
    # If it does not exist, create it
    os.makedirs(zero_shot_with_judge_scaling_output_dir)
titles_1_1 = ['Claude-3-haiku - zero_shot_with_judge', 'Claude-3-haiku - zero_shot_with_judge']
generate_statistics_and_plots_2(zero_shot_with_judge_scaling_list_for_claude, sample_size_list_2, cols, zero_shot_with_judge_scaling_output_dir, titles_1_1)
titles_1_2 = ['GPT-4o-mini - zero_shot_with_judge', 'GPT-4o-mini - zero_shot_with_judge']
generate_statistics_and_plots_2(zero_shot_with_judge_scaling_list_for_gpt, sample_size_list_2, cols, zero_shot_with_judge_scaling_output_dir, titles_1_2)

zero_shot_without_judge_scaling_output_dir = f"/home/jlong1/Downloads/persuasion/Data_n_Code_persuasion/jikailoong/3_evaluation_results/{args.book}/{args.technique_dir}/statistical_report/inference_scaling/zero_shot_without_judge"
if not os.path.exists(zero_shot_without_judge_scaling_output_dir):
    # If it does not exist, create it
    os.makedirs(zero_shot_without_judge_scaling_output_dir)
titles_2_1 = ['Claude-3-haiku - zero_shot_without_judge', 'Claude-3-haiku - zero_shot_without_judge']
generate_statistics_and_plots_2(zero_shot_without_judge_scaling_list_for_claude, sample_size_list_2, cols, zero_shot_without_judge_scaling_output_dir, titles_2_1)
titles_2_2 = ['GPT-4o-mini - zero_shot_without_judge', 'GPT-4o-mini - zero_shot_without_judge']
generate_statistics_and_plots_2(zero_shot_without_judge_scaling_list_for_gpt, sample_size_list_2, cols, zero_shot_without_judge_scaling_output_dir, titles_2_2)

few_shots_with_judge_scaling_output_dir = f"/home/jlong1/Downloads/persuasion/Data_n_Code_persuasion/jikailoong/3_evaluation_results/{args.book}/{args.technique_dir}/statistical_report/inference_scaling/few_shots_with_judge"
if not os.path.exists(few_shots_with_judge_scaling_output_dir):
    # If it does not exist, create it
    os.makedirs(few_shots_with_judge_scaling_output_dir)
titles_3_1 = ['Claude-3-haiku - few_shots_with_judge', 'Claude-3-haiku - few_shots_with_judge']
generate_statistics_and_plots_2(few_shots_with_judge_scaling_list_for_claude, sample_size_list_2, cols, few_shots_with_judge_scaling_output_dir, titles_3_1)
titles_3_2 = ['GPT-4o-mini - few_shots_with_judge', 'GPT-4o-mini - few_shots_with_judge']
generate_statistics_and_plots_2(few_shots_with_judge_scaling_list_for_gpt, sample_size_list_2, cols, few_shots_with_judge_scaling_output_dir, titles_3_2)

few_shots_without_judge_scaling_output_dir = f"/home/jlong1/Downloads/persuasion/Data_n_Code_persuasion/jikailoong/3_evaluation_results/{args.book}/{args.technique_dir}/statistical_report/inference_scaling/few_shots_without_judge"
if not os.path.exists(few_shots_without_judge_scaling_output_dir):
    # If it does not exist, create it
    os.makedirs(few_shots_without_judge_scaling_output_dir)
titles_4_1 = ['Claude-3-haiku - few_shots_without_judge', 'Claude-3-haiku - few_shots_without_judge']
generate_statistics_and_plots_2(few_shots_without_judge_scaling_list_for_claude, sample_size_list_2, cols, few_shots_without_judge_scaling_output_dir, titles_4_1)
titles_4_2 = ['GPT-4o-mini - few_shots_without_judge', 'GPT-4o-mini - few_shots_without_judge']
generate_statistics_and_plots_2(few_shots_without_judge_scaling_list_for_gpt, sample_size_list_2, cols, few_shots_without_judge_scaling_output_dir, titles_4_2)
