import pandas as pd
import numpy as np
from pathlib import Path

# ==================== 配置 ====================
INPUT_DIR = r"E:\统计建模\统计建模2026\data\省市\省市"
OUTPUT_DIR = r"E:\统计建模\统计建模2026\data\处理后指标"
YEAR_START, YEAR_END = 2016, 2024
YEARS = list(range(YEAR_START, YEAR_END + 1))

ORIGINAL_INDICATORS = [
    "制造业城镇单位就业人员 (万人)",
    "制造业城镇单位就业人员平均工资 (元)",
    "人均地区生产总值 (元/人)",
    "固定资产投资价格指数 (上年=100)",
    "65岁及以上人口数 (人口抽样调查) (人)",
    "15-64岁人口数 (人口抽样调查) (人)",
    "老年人口抚养比 (人口抽样调查) (%)",
    "城乡居民基本养老保险基金支出 (亿元)",
    "城镇居民人均可支配收入 (元)",
    "医院数 (个)",
    "建设工程监理企业从业人数 (人)",
    "电力消费量 (亿千瓦小时)",
    "发电量 (亿千瓦时)",
    "建筑业企业利润总额 (亿元)",
    "道路面积 (万平方米)",
    "房地产开发企业竣工房屋面积 (万平方米)",
    "地区生产总值 (亿元)",
    "第二产业增加值 (亿元)",
    "建筑业私营企业和个体就业人员 (万人)",
    "生产安全事故死亡人数（人）"
]

# 新指标及计算公式
NEW_INDICATORS_DEF = {
    "制造业就业基数": "制造业城镇单位就业人员 (万人)",
    "人力成本当量": lambda df: df["制造业城镇单位就业人员平均工资 (元)"] / 12,
    "岗位产出效率": lambda df: df["第二产业增加值 (亿元)"] / df["制造业城镇单位就业人员 (万人)"],
    "投资门槛指数": "固定资产投资价格指数 (上年=100)",
    "人均地区生产总值": "人均地区生产总值 (元/人)",
    "制造业经济权重": lambda df: df["第二产业增加值 (亿元)"] / df["地区生产总值 (亿元)"] * 100,
    "老年抚养比": "老年人口抚养比 (人口抽样调查) (%)",
    "高龄人口覆盖比": lambda df: df["65岁及以上人口数 (人口抽样调查) (人)"] /
                                 (df["65岁及以上人口数 (人口抽样调查) (人)"] +
                                  df["15-64岁人口数 (人口抽样调查) (人)"]) * 100,
    "公共养老财政负荷": "城乡居民基本养老保险基金支出 (亿元)",
    "居民消费潜力": "城镇居民人均可支配收入 (元)",
    "千名老人医疗机构稀缺性": lambda df: df["医院数 (个)"] /
                                         df["65岁及以上人口数 (人口抽样调查) (人)"] * 1000,
    "生产安全事故死亡人数": "生产安全事故死亡人数（人）",
    "危险岗位从业人员": "建筑业私营企业和个体就业人员 (万人)",
    "电力系统维护当量": "电力消费量 (亿千瓦小时)",
    "路网养护规模": "道路面积 (万平方米)",
    "房屋安全存量": "房地产开发企业竣工房屋面积 (万平方米)",
    "建筑业企业利润总额": "建筑业企业利润总额 (亿元)",
    "电力产业规模": "发电量 (亿千瓦时)"
}

# 需要特殊处理的指标（整列数据起始年份晚于 YEAR_START 时不向前填充）
SKIP_FORWARD_FILL_COLS = [
    "生产安全事故死亡人数（人）"
]

# ==================== 函数 ====================
def read_province_data(file_path):
    """读取单个省份数据，返回年份×原始指标的DataFrame"""
    df_raw = pd.read_excel(file_path, header=3, index_col=0)
    if "2025" in df_raw.columns:
        df_raw = df_raw.drop(columns=["2025"])
    df = df_raw.T
    df.columns = df.columns.str.strip()
    df.index = df.index.astype(str).str.replace("年", "", regex=False).astype(int)
    # 保留存在的 ORIGINAL_INDICATORS，缺失的后面会补全
    df = df[[col for col in ORIGINAL_INDICATORS if col in df.columns]]
    df = df.loc[YEARS]
    return df

def fill_missing(series, national_avg_series):
    """
    缺失值填充策略：
    - 小范围缺失（连续缺失 < 3 年）: 取前后平均值（若只有一侧则复制该侧）
    - 大范围缺失（连续缺失 ≥ 3 年）: 用本省所有有效年份的均值
    - 完全无数据: 用全国同年平均值，最后用全国全局均值兜底
    """
    series = series.copy()
    orig = series.copy()
    all_valid = orig.dropna()
    prov_avg = all_valid.mean() if len(all_valid) > 0 else np.nan

    missing_years = [y for y in series.index if pd.isna(series.loc[y])]
    gaps = []
    if missing_years:
        sorted_missing = sorted(missing_years)
        start = sorted_missing[0]
        end = sorted_missing[0]
        for y in sorted_missing[1:]:
            if y == end + 1:
                end = y
            else:
                gaps.append((start, end))
                start = y
                end = y
        gaps.append((start, end))

    for gap_start, gap_end in gaps:
        gap_len = gap_end - gap_start + 1
        if gap_len < 3:
            for y in range(gap_start, gap_end + 1):
                prev_val = next_val = None
                for yp in range(y - 1, series.index.min() - 1, -1):
                    if yp in orig.index and pd.notna(orig.loc[yp]):
                        prev_val = orig.loc[yp]
                        break
                for yn in range(y + 1, series.index.max() + 1):
                    if yn in orig.index and pd.notna(orig.loc[yn]):
                        next_val = orig.loc[yn]
                        break
                if prev_val is not None and next_val is not None:
                    series.loc[y] = (prev_val + next_val) / 2
                elif prev_val is not None:
                    series.loc[y] = prev_val
                elif next_val is not None:
                    series.loc[y] = next_val
        else:
            if pd.notna(prov_avg):
                series.loc[gap_start:gap_end] = prov_avg
            else:
                for y in range(gap_start, gap_end + 1):
                    if y in national_avg_series.index and pd.notna(national_avg_series.loc[y]):
                        series.loc[y] = national_avg_series.loc[y]
                    else:
                        global_avg = national_avg_series.mean()
                        series.loc[y] = global_avg if pd.notna(global_avg) else 0.0

    # 兜底：填充所有剩余缺失
    for year in series.index:
        if pd.isna(series.loc[year]):
            if year in national_avg_series.index and pd.notna(national_avg_series.loc[year]):
                series.loc[year] = national_avg_series.loc[year]
            else:
                global_avg = national_avg_series.mean()
                series.loc[year] = global_avg if pd.notna(global_avg) else 0.0
    return series

def compute_national_averages(all_data):
    """计算全国各年各指标的平均值（基于原始数据）"""
    records = []
    for prov, df in all_data.items():
        for year in df.index:
            for col in df.columns:
                val = df.loc[year, col]
                if pd.notna(val):
                    records.append({"year": year, "indicator": col, "value": val})
    if not records:
        return pd.DataFrame(index=YEARS, columns=ORIGINAL_INDICATORS)
    temp = pd.DataFrame(records)
    national_avg = temp.groupby(["year", "indicator"])["value"].mean().unstack()
    national_avg = national_avg.reindex(YEARS)
    return national_avg

def main():
    input_path = Path(INPUT_DIR)
    output_path = Path(OUTPUT_DIR)
    output_path.mkdir(parents=True, exist_ok=True)

    # 读取所有省份原始数据
    all_data = {}
    for file in input_path.glob("*.xlsx"):
        if file.name.startswith("~$"):
            continue
        try:
            all_data[file.stem] = read_province_data(file)
            print(f"读取成功: {file.stem}")
        except Exception as e:
            print(f"读取失败 {file.name}: {e}")

    if not all_data:
        raise RuntimeError("未读取到任何有效数据，请检查路径。")

    # 计算全国平均值
    national_avg_df = compute_national_averages(all_data)
    # 补全全国平均值中缺失的指标列（例如完全无数据的列）
    for col in ORIGINAL_INDICATORS:
        if col not in national_avg_df.columns:
            national_avg_df[col] = np.nan
    print("全国平均值计算完成。")

    # 处理每个省份
    for prov, df_orig in all_data.items():
        df = df_orig.copy()

        # ---------- 1. 补全当前省份完全缺失的指标列（全为 NaN）----------
        for col in ORIGINAL_INDICATORS:
            if col not in df.columns:
                df[col] = np.nan

        # ---------- 2. 特殊处理：列数据起始年份晚于 YEAR_START 时的向前填充 ----------
        for col in df.columns:
            valid_years = [y for y in df.index if pd.notna(df.loc[y, col])]
            if not valid_years:
                continue
            if min(valid_years) > YEAR_START:
                if col in SKIP_FORWARD_FILL_COLS:
                    continue
                first_valid_mean = df.loc[valid_years, col].mean()
                for y in range(YEAR_START, min(valid_years)):
                    df.loc[y, col] = first_valid_mean

        # ---------- 3. 对所有原始指标执行缺失值填充 ----------
        for col in ORIGINAL_INDICATORS:
            if col not in df.columns:
                continue
            national_avg_series = national_avg_df[col]  # 现在一定存在
            df[col] = fill_missing(df[col], national_avg_series)

        # ---------- 4. 计算新指标 ----------
        result = pd.DataFrame(index=df.index)
        for new_name, formula in NEW_INDICATORS_DEF.items():
            try:
                if isinstance(formula, str):
                    if formula in df.columns:
                        result[new_name] = df[formula]
                    else:
                        result[new_name] = np.nan
                else:
                    result[new_name] = formula(df)
            except Exception as e:
                print(f"计算 {new_name} 出错 ({prov}): {e}")
                result[new_name] = np.nan

        # ---------- 5. 保存结果 ----------
        out_file = output_path / f"{prov}.xlsx"
        result.to_excel(out_file, index_label="年份")
        print(f"已保存: {out_file}")

    print("全部处理完成。")

if __name__ == "__main__":
    main()