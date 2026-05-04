import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import gaussian_kde
from itertools import product

matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

# ================== 参数设置 ==================
file_path = r"E:\统计建模\统计建模2026\data\综合紧迫度结果.xlsx"
sheet_names = ["制造业", "养老产业", "特殊危险环境产业"]
year_start = 2016
year_end = 2024 
years = list(range(year_start, year_end + 1))

# Markov 等级划分边界（0~1 四等分）
bins = [0, 0.25, 0.5, 0.75, 1.0]
labels = ["I (低紧迫)", "II (中低紧迫)", "III (中高紧迫)", "IV (高紧迫)"]

# ================== 读取数据 ==================
def load_industry_data(sheet_name):
    df = pd.read_excel(file_path, sheet_name=sheet_name, index_col=0)
    # 确保年份列为字符串转整数
    df.columns = df.columns.astype(int)
    # 只保留指定年份列
    df = df[[col for col in years if col in df.columns]]
    return df

# ================== 核密度估计绘图 ==================
def plot_kde_evolution(df, industry_name, years):
    """绘制逐年核密度曲线叠加图"""
    plt.figure(figsize=(10, 6))
    colors = sns.color_palette("viridis", len(years))
    for i, year in enumerate(years):
        if year not in df.columns:
            continue
        scores = df[year].dropna().values
        if len(scores) < 2:
            continue
        kde = gaussian_kde(scores, bw_method='scott')
        x_grid = np.linspace(0, 1, 200)
        density = kde(x_grid)
        plt.plot(x_grid, density, color=colors[i], label=str(year), linewidth=2)
    plt.xlabel("综合紧迫度得分 U", fontsize=12)
    plt.ylabel("核密度", fontsize=12)
    plt.title(f"{industry_name} 核密度曲线演变", fontsize=14)
    plt.legend(title="年份")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(rf"E:\统计建模\统计建模2026\fig\{industry_name}_核密度曲线.png", dpi=300)
    plt.show()

# ================== Markov 链分析 ==================
def discretize_scores(scores, bins, labels):
    """将得分转化为等级标签"""
    return pd.cut(scores, bins=bins, labels=labels, include_lowest=True)

def build_transition_matrix(df, years, bins, labels):
    """构建一步转移概率矩阵（基于相邻年份）"""
    n_states = len(labels)
    trans_count = np.zeros((n_states, n_states))
    # 遍历所有省份和相邻年份
    for province in df.index:
        for t in range(len(years)-1):
            year_cur = years[t]
            year_next = years[t+1]
            if year_cur not in df.columns or year_next not in df.columns:
                continue
            s_cur = df.loc[province, year_cur]
            s_next = df.loc[province, year_next]
            if pd.isna(s_cur) or pd.isna(s_next):
                continue
            cur_label = discretize_scores(pd.Series([s_cur]), bins, labels)[0]
            next_label = discretize_scores(pd.Series([s_next]), bins, labels)[0]
            i = labels.index(cur_label)
            j = labels.index(next_label)
            trans_count[i, j] += 1
    # 归一化得到概率矩阵
    trans_prob = trans_count / trans_count.sum(axis=1, keepdims=True)
    trans_prob = np.nan_to_num(trans_prob)  # 处理全零行
    return trans_prob, trans_count

def plot_transition_matrix(trans_prob, labels, industry_name):
    """绘制转移概率热力图"""
    plt.figure(figsize=(8, 6))
    sns.heatmap(trans_prob, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=labels, yticklabels=labels, cbar_kws={'label': '转移概率'})
    plt.xlabel("t+1 状态", fontsize=12)
    plt.ylabel("t 状态", fontsize=12)
    plt.title(f"{industry_name} 一步转移概率矩阵", fontsize=14)
    plt.tight_layout()
    plt.savefig(rf"E:\统计建模\统计建模2026\fig\{industry_name}_Markov转移矩阵.png", dpi=300)
    plt.show()

def multi_step_transition(trans_prob, steps):
    """计算多步转移概率矩阵（幂乘法）"""
    result = trans_prob.copy()
    for _ in range(steps-1):
        result = np.dot(result, trans_prob)
    return result

# ================== 主程序 ==================
if __name__ == "__main__":
    for sheet in sheet_names:
        print(f"\n===== 正在处理：{sheet} =====")
        df = load_industry_data(sheet)
        if df.empty:
            print(f"警告：{sheet} 无有效数据")
            continue
        
        # 1. 核密度估计
        plot_kde_evolution(df, sheet, years)
        
        # 2. Markov 链分析
        trans_prob, trans_count = build_transition_matrix(df, years, bins, labels)
        print(f"一步转移概率矩阵：\n{pd.DataFrame(trans_prob, index=labels, columns=labels)}\n")
        plot_transition_matrix(trans_prob, labels, sheet)
        
        # 可选：输出多步转移概率（例如滞后2~5年）
        for step in [2, 3, 4, 5]:
            multi_prob = multi_step_transition(trans_prob, step)
            print(f"滞后{step}年转移概率矩阵示例（前两行）：\n{multi_prob[:2]}\n")
            # 可将重要结果保存为表格供论文使用
        
        # 将一步转移矩阵保存为 CSV（方便导入论文表格）
        pd.DataFrame(trans_prob, index=labels, columns=labels).to_csv(rf"E:\统计建模\统计建模2026\data\{sheet}_一步转移矩阵.csv")
        
        # 额外输出各等级样本数分布（可用于论文说明）
        all_scores = df.values.flatten()
        all_scores = all_scores[~np.isnan(all_scores)]
        grade_dist = discretize_scores(pd.Series(all_scores), bins, labels).value_counts().sort_index()
        print("等级样本分布：\n", grade_dist)