import pandas as pd
import numpy as np
import os

# ============ 配置区域 ============
DATA_DIR = r"E:\统计建模\统计建模2026\data\处理后指标"          # 存放各省份 xlsx 文件的目录
OUTPUT_FILE = r"E:\统计建模\统计建模2026\data\综合紧迫度结果.xlsx"    # 输出文件名
P_INDUSTRY = {
    "制造业": 0.2787,
    "养老产业": 0.2719,
    "特殊危险环境产业": 0.4494
}
ALPHA = 0.6   # 数据驱动权重
BETA = 0.4    # 产业特质权重
# ================================

# 1. 定义指标与产业、维度的映射关系
industry_indicators = {
    "制造业": {
        "人力替代压力": ["制造业就业基数", "人力成本当量"],
        "技术经济可行性": ["岗位产出效率", "投资门槛指数"],
        "区域发展底座": ["人均地区生产总值", "制造业经济权重"]
    },
    "养老产业": {
        "人口老龄化压力": ["老年抚养比", "高龄人口覆盖比"],
        "社会支付能力": ["公共养老财政负荷", "居民消费潜力"],
        "养老资源稀缺度": ["千名老人医疗机构稀缺性"]
    },
    "特殊危险环境产业": {
        "作业风险与人力规模": ["生产安全事故死亡人数", "危险岗位从业人员"],
        "设施维护体量": ["电力系统维护当量", "路网养护规模", "房屋安全存量"],
        "产业支撑强度": ["建筑业企业利润总额", "电力产业规模"]
    }
}

# 所有负向指标（需要正向化）
negative_indicators = ["投资门槛指数", "千名老人医疗机构稀缺性"]

# 所有正向指标（其余默认正向）
all_indicators = []
for dims in industry_indicators.values():
    for inds in dims.values():
        all_indicators.extend(inds)
all_indicators = list(set(all_indicators))

# 2. 读取并合并所有省份数据
def load_data(data_dir):
    records = []
    for fname in os.listdir(data_dir):
        if fname.endswith(".xlsx") and not fname.startswith("~$"):
            province = fname.replace(".xlsx", "")
            df = pd.read_excel(os.path.join(data_dir, fname))
            # 确保第一列是年份，且列名统一
            year_col = df.columns[0]
            df.rename(columns={year_col: "年份"}, inplace=True)
            df["省份"] = province
            records.append(df)
    full_df = pd.concat(records, ignore_index=True)
    return full_df

print("正在读取数据...")
data = load_data(DATA_DIR)

# 3. 数据预处理：指标正向化与全局极差归一化
def preprocess_indicators(df, indicators, neg_indicators):
    df_norm = df[["省份", "年份"]].copy()
    for col in indicators:
        if col not in df.columns:
            raise ValueError(f"指标列 '{col}' 在数据中不存在")
        # 正向化
        if col in neg_indicators:
            # 负向指标：max - x
            pos_vals = df[col].max() - df[col]
        else:
            pos_vals = df[col].copy()
        # 全局极差归一化
        min_val = pos_vals.min()
        max_val = pos_vals.max()
        if max_val - min_val == 0:
            norm_vals = 0.0
        else:
            norm_vals = (pos_vals - min_val) / (max_val - min_val)
        df_norm[col] = norm_vals
    return df_norm

df_norm = preprocess_indicators(data, all_indicators, negative_indicators)

# 4. 工具函数：熵权法计算权重
def entropy_weights(matrix):
    """
    matrix: numpy 2-D array, shape (n_samples, n_features)
    返回: 权重 array of shape (n_features,)
    """
    # 避免除以0
    matrix = matrix + 1e-12
    col_sums = matrix.sum(axis=0)
    p = matrix / col_sums
    # 处理 log(0) 情况
    p = np.where(p == 0, 1e-12, p)
    n_samples = matrix.shape[0]
    e = -np.sum(p * np.log(p), axis=0) / np.log(n_samples)
    d = 1 - e
    w = d / d.sum()
    return w

# 5. 工具函数：TOPSIS 聚合（按年份独立确定理想解）
def topsis_aggregate_by_year(df_long, indicators, weights, year_col="年份", prov_col="省份"):
    """
    df_long: DataFrame，包含 省份、年份 以及各指标列（已归一化）
    indicators: 该维度下的指标名列表
    weights: 指标权重数组（长度与 indicators 一致）
    返回: DataFrame，包含 省份、年份、得分
    """
    # 加权标准化矩阵
    val_cols = df_long[indicators].values
    weighted = val_cols * weights[np.newaxis, :]  # 广播乘法
    
    years = df_long[year_col].unique()
    result_list = []
    for yr in years:
        idx = df_long[year_col] == yr
        vy = weighted[idx]  # n_prov * m
        provs = df_long.loc[idx, prov_col].values
        # 理想解（每年独立）
        ideal_pos = np.max(vy, axis=0)
        ideal_neg = np.min(vy, axis=0)
        # 欧氏距离
        d_pos = np.sqrt(np.sum((vy - ideal_pos) ** 2, axis=1))
        d_neg = np.sqrt(np.sum((vy - ideal_neg) ** 2, axis=1))
        # 相对贴近度
        with np.errstate(divide='ignore', invalid='ignore'):
            score = d_neg / (d_pos + d_neg)
            score = np.where(np.isnan(score), 0.5, score)  # 当两距离均为0时给0.5
        tmp = pd.DataFrame({
            prov_col: provs,
            year_col: yr,
            "score": score
        })
        result_list.append(tmp)
    return pd.concat(result_list, ignore_index=True)


# 6. 计算单个产业的“数据驱动得分 S_data”
def compute_S_data(df_norm, industry_dims, year_col="年份", prov_col="省份"):
    """
    df_norm: 归一化后的全量数据 DataFrame
    industry_dims: 字典 {维度名: [指标列表]}
    返回: DataFrame，包含 省份、年份、S_data 以及各维度得分
    """
    # 准备收集维度得分矩阵 C^(d)
    dim_scores = {}   # dim_name -> DataFrame with 省份, 年份, score
    # 第一层：维度得分
    for dim, inds in industry_dims.items():
        # 筛选数据
        sub_df = df_norm[[prov_col, year_col] + inds].dropna()
        # 构建用于熵权法的全样本矩阵 (nT * m_d)
        matrix = sub_df[inds].values
        w = entropy_weights(matrix)
        # TOPSIS 聚合得维度得分
        dim_score_df = topsis_aggregate_by_year(sub_df, inds, w, year_col, prov_col)
        dim_score_df.rename(columns={"score": dim}, inplace=True)
        dim_scores[dim] = dim_score_df
    
    # 合并所有维度得分为一个大表
    merged = dim_scores[list(industry_dims.keys())[0]]
    for dim in list(industry_dims.keys())[1:]:
        merged = merged.merge(dim_scores[dim], on=[prov_col, year_col], how="inner")
    
    dim_names = list(industry_dims.keys())
    # 第二层：维度权重计算（全样本）
    C_matrix = merged[dim_names].values  # (n*T) x D
    lambda_w = entropy_weights(C_matrix)
    
    # 加权矩阵
    weighted_C = C_matrix * lambda_w[np.newaxis, :]
    # TOPSIS 按年聚合
    result_rows = []
    for yr in merged[year_col].unique():
        idx = merged[year_col] == yr
        vy = weighted_C[idx]
        provs = merged.loc[idx, prov_col].values
        ideal_pos = np.max(vy, axis=0)
        ideal_neg = np.min(vy, axis=0)
        d_pos = np.sqrt(np.sum((vy - ideal_pos) ** 2, axis=1))
        d_neg = np.sqrt(np.sum((vy - ideal_neg) ** 2, axis=1))
        with np.errstate(divide='ignore', invalid='ignore'):
            score = d_neg / (d_pos + d_neg)
            score = np.where(np.isnan(score), 0.5, score)
        for j, prov in enumerate(provs):
            result_rows.append({prov_col: prov, year_col: yr, "S_data": score[j]})
    S_df = pd.DataFrame(result_rows)
    return S_df

# 7. 计算所有产业的数据驱动得分并融合 P_industry
print("开始计算产业得分...")
output_workbook = {}

for industry, dims in industry_indicators.items():
    print(f"  处理 {industry} ...")
    S_df = compute_S_data(df_norm, dims)
    # 融合 P_industry
    p_val = P_INDUSTRY[industry]
    S_df["U"] = (S_df["S_data"] ** ALPHA) * (p_val ** BETA)
    # 整理为省份×年份透视表
    pivot = S_df.pivot(index="省份", columns="年份", values="U")
    # 确保年份顺序
    years_sorted = sorted(pivot.columns, reverse=False)
    pivot = pivot[years_sorted]
    output_workbook[industry] = pivot

# 8. 写入最终 xlsx 文件（每个产业一个 sheet）
print("正在导出结果...")
with pd.ExcelWriter(OUTPUT_FILE) as writer:
    for industry, pivot in output_workbook.items():
        # 使用简洁的 sheet 名（不超过31字符）
        sheet_name = industry if len(industry) <= 31 else industry[:28] + "..."
        pivot.to_excel(writer, sheet_name=sheet_name, index=True)

print(f"计算完成！结果已保存至：{OUTPUT_FILE}")
print("文件包含三个工作表：制造业、养老产业、特殊危险环境产业")
print("每个工作表行索引为省份，列为年份，单元格值为综合紧迫度 U")