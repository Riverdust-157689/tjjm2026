import pandas as pd
from pathlib import Path

INPUT_DIR = r"E:\统计建模\统计建模2026\data\省市\省市"
OUTPUT_DIR = r"E:\统计建模\统计建模2026\data\处理后指标"
YEAR_START, YEAR_END = 2016, 2024
YEARS = list(range(YEAR_START, YEAR_END + 1))
COL_NAME_RAW = "生产安全事故死亡人数（人）"   # 原始文件列名（全角括号）
COL_NAME_OUT = "生产安全事故死亡人数"        # 输出文件列名


def read_raw_col(file_path):
    """从原始文件读取指定列，转置后以年份为索引"""
    df_raw = pd.read_excel(file_path, header=3, index_col=0)
    if "2025" in df_raw.columns:
        df_raw = df_raw.drop(columns=["2025"])
    df = df_raw.T
    df.columns = df.columns.str.strip()
    df.index = df.index.astype(str).str.replace("年", "", regex=False).astype(int)
    if COL_NAME_RAW in df.columns:
        return df[COL_NAME_RAW].loc[YEARS]
    return pd.Series(dtype=float, index=YEARS)


def main():
    input_path = Path(INPUT_DIR)
    output_path = Path(OUTPUT_DIR)

    for out_file in output_path.glob("*.xlsx"):
        prov = out_file.stem
        raw_file = input_path / f"{prov}.xlsx"

        # 读取原始数据中的该列
        raw_series = read_raw_col(raw_file)

        # 读取输出文件
        df = pd.read_excel(out_file, index_col=0)

        # 已有数据（2021-2024）的均值
        valid = raw_series.dropna()
        if len(valid) > 0:
            prov_avg = valid.mean()
        else:
            prov_avg = 0.0

        # 填充 2016-2020：缺失的用均值
        for y in range(YEAR_START, YEAR_END + 1):
            if y in df.index:
                if pd.isna(df.at[y, COL_NAME_OUT]) or df.at[y, COL_NAME_OUT] == 0:
                    df.at[y, COL_NAME_OUT] = prov_avg

        df.to_excel(out_file)
        print(f"已修复: {prov}, 填充值={prov_avg:.2f}")

    print("全部完成。")


if __name__ == "__main__":
    main()