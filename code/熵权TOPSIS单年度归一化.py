import pandas as pd
import numpy as np
import os

ALPHA = 0.6
BETA = 0.4
prefix = f"A{ALPHA}B{BETA}"
# ============ 配置区域 ============
DATA_DIR = r"E:\统计建模\统计建模2026\data\处理后指标"
OUTPUT_FILE = rf"E:\统计建模\统计建模2026\data\综合紧迫度结果_修正{prefix}.xlsx"

P_INDUSTRY = {
    "制造业": 0.2787,
    "养老产业": 0.2719,
    "特殊危险环境产业": 0.4494
}

# =================================


# 指标体系（保持不变）
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

negative_indicators = ["投资门槛指数", "千名老人医疗机构稀缺性"]

# 省份输出顺序（按区域）
# PROVINCE_ORDER = [
#     "北京", "天津", "河北", "上海", "江苏", "浙江", "福建", "山东", "广东", "海南",
#     "山西", "安徽", "江西", "河南", "湖北", "湖南",
#     "内蒙古", "广西", "四川", "重庆", "贵州", "云南", "陕西", "甘肃", "青海", "宁夏", "新疆",
#     "辽宁", "吉林", "黑龙江"
# ]
PROVINCE_ORDER = [
"北京", "天津", "河北", "山西", "内蒙古", "辽宁", "吉林", "黑龙江",
"广东", "广西", "海南",
"上海", "江苏", "浙江", "福建", "山东",
"安徽", "江西", "河南", "湖北", "湖南",
"重庆", "四川", "贵州", "云南", "西藏", "陕西", "甘肃", "青海", "宁夏", "新疆"
]
# ===============================
# 1. 读取数据
# ===============================
def load_data(data_dir):
    records = []
    for fname in os.listdir(data_dir):
        if fname.endswith(".xlsx") and not fname.startswith("~$"):
            province = fname.replace(".xlsx", "")
            df = pd.read_excel(os.path.join(data_dir, fname))
            year_col = df.columns[0]
            df.rename(columns={year_col: "年份"}, inplace=True)
            df["省份"] = province
            records.append(df)
    return pd.concat(records, ignore_index=True)


# ===============================
# 2. 按年份归一化（关键修正）
# ===============================
def normalize_by_year(df, indicators, neg_inds):
    result = []

    for yr in df["年份"].unique():
        sub = df[df["年份"] == yr].copy()

        for col in indicators:
            if col not in sub.columns:
                raise ValueError(f"{col} 不存在")

            # 缺失值填补（均值）
            sub[col] = sub[col].fillna(sub[col].mean())

            min_val = sub[col].min()
            max_val = sub[col].max()

            if max_val - min_val == 0:
                sub[col] = 0.5
            else:
                if col in neg_inds:
                    sub[col] = (max_val - sub[col]) / (max_val - min_val)
                else:
                    sub[col] = (sub[col] - min_val) / (max_val - min_val)

        result.append(sub)

    return pd.concat(result, ignore_index=True)


# ===============================
# 3. 熵权法（单年份）
# ===============================
def entropy_weight(matrix):
    matrix = matrix + 1e-12
    p = matrix / matrix.sum(axis=0)

    n = matrix.shape[0]
    e = -np.sum(p * np.log(p), axis=0) / np.log(n)
    d = 1 - e
    w = d / d.sum()
    return w


# ===============================
# 4. TOPSIS（单年份）
# ===============================
def topsis(matrix, weights):
    weighted = matrix * weights

    ideal_pos = np.max(weighted, axis=0)
    ideal_neg = np.min(weighted, axis=0)

    d_pos = np.sqrt(((weighted - ideal_pos) ** 2).sum(axis=1))
    d_neg = np.sqrt(((weighted - ideal_neg) ** 2).sum(axis=1))

    score = d_neg / (d_pos + d_neg)
    return np.nan_to_num(score, nan=0.5)


# ===============================
# 5. 核心：计算产业得分（完全重写）
# ===============================
def compute_S_data(df_norm, dims):
    all_results = []

    for yr in df_norm["年份"].unique():
        year_df = df_norm[df_norm["年份"] == yr].copy()

        dim_scores = {}

        # ---------- 第一层：维度 ----------
        for dim, inds in dims.items():
            X = year_df[inds].values

            w = entropy_weight(X)
            s = topsis(X, w)

            dim_scores[dim] = s

        # 拼接维度矩阵
        C = np.column_stack(list(dim_scores.values()))

        # ---------- 第二层：维度权重 ----------
        w_dim = entropy_weight(C)
        final_score = topsis(C, w_dim)

        tmp = pd.DataFrame({
            "省份": year_df["省份"].values,
            "年份": yr,
            "S_data": final_score
        })

        all_results.append(tmp)

    return pd.concat(all_results, ignore_index=True)


# ===============================
# 主流程
# ===============================
print("读取数据...")
data = load_data(DATA_DIR)

# 收集指标
all_inds = []
for d in industry_indicators.values():
    for inds in d.values():
        all_inds.extend(inds)
all_inds = list(set(all_inds))

print("归一化处理...")
df_norm = normalize_by_year(data, all_inds, negative_indicators)

# ... 前面保持不变 ...

print("计算结果...")
long_list = []  # 用于收集所有产业的长表

for industry, dims in industry_indicators.items():
    print(f"处理 {industry}")

    S_df = compute_S_data(df_norm, dims)

    p = P_INDUSTRY[industry]
    S_df["U"] = (S_df["S_data"] ** ALPHA) * (p ** BETA)
    S_df["产业"] = industry  # 标记产业

    long_list.append(S_df[["省份", "年份", "产业", "U"]])

# 合并为一张长表
all_long = pd.concat(long_list, ignore_index=True)

# 省内同年归一化：U_share = U / sum(U) （同一省份、同一年份的三个产业）
all_long["U_sum"] = all_long.groupby(["省份", "年份"])["U"].transform("sum")
# 避免除以零（理论上sum>0，因为非负且至少有一个正数）
all_long["U_share"] = np.where(all_long["U_sum"] > 0,
                               all_long["U"] / all_long["U_sum"],
                               1/3)  # 极端情况平均分配

# 准备输出：U 和 U_share 分别透视
output_u = {}
output_share = {}

for industry in industry_indicators.keys():
    sub = all_long[all_long["产业"] == industry].copy()

    # U 透视表
    pivot_u = sub.pivot(index="省份", columns="年份", values="U")
    pivot_u = pivot_u[sorted(pivot_u.columns)]
    pivot_u = pivot_u.reindex(PROVINCE_ORDER)
    output_u[industry] = pivot_u

    # U_share 透视表
    pivot_share = sub.pivot(index="省份", columns="年份", values="U_share")
    pivot_share = pivot_share[sorted(pivot_share.columns)]
    pivot_share = pivot_share.reindex(PROVINCE_ORDER)
    output_share[industry + "_份额"] = pivot_share

print("导出Excel...")
with pd.ExcelWriter(OUTPUT_FILE) as writer:
    # 原始 U 值
    for k, v in output_u.items():
        v.to_excel(writer, sheet_name=k)
    # 省内归一化份额
    for k, v in output_share.items():
        v.to_excel(writer, sheet_name=k)

print("✅ 完成（含省内同年归一化份额）")