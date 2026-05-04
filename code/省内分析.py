import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go  # 可选，用于交互式三分图
import warnings
warnings.filterwarnings('ignore')

matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

# ================== 参数设置 ==================
file_path = r"E:\统计建模\统计建模2026\data\综合紧迫度结果.xlsx"
sheet_names = ["制造业", "养老产业", "特殊危险环境产业"]
year_start = 2016
year_end = 2024 
years = list(range(year_start, year_end + 1))

# ================== 数据读取与融合 ==================
def load_all_data():
    """读取三个产业数据，融合成一个 MultiIndex DataFrame：省份-年份，三列得分"""
    data_dict = {}
    for sheet in sheet_names:
        df = pd.read_excel(file_path, sheet_name=sheet, index_col=0)
        df.columns = [int(c) if str(c).replace('.','').isdigit() else c for c in df.columns]
        df = df[years]  # 只保留目标年份
        data_dict[sheet] = df.stack().rename(sheet)  # 堆叠成 Series
    # 合并三个产业
    combined = pd.DataFrame(data_dict)
    combined.index.names = ['省份', '年份']
    combined = combined.reset_index()
    # 将年份转换为整数
    combined['年份'] = combined['年份'].astype(int)
    return combined

df_all = load_all_data()
print("数据预览：")
print(df_all.head())

# ================== 1. 省内领先产业识别与省域类型划分 ==================
def lead_industry_analysis(df):
    """识别每个省份每年领先产业，统计主导类型"""
    industries = sheet_names
    # 找出最大值所在的产业
    df['领先产业'] = df[industries].idxmax(axis=1)
    # 处理并列最大值：取第一个出现的产业（可改为"混合"）
    # 为了避免并列，可判断最大值是否唯一
    max_vals = df[industries].max(axis=1)
    is_tie = (df[industries].eq(max_vals, axis=0).sum(axis=1) > 1)
    df.loc[is_tie, '领先产业'] = '混合领先'
    
    # 按省份统计领先产业频次
    leading_counts = df.groupby(['省份', '领先产业']).size().unstack(fill_value=0)
    leading_counts['主导类型'] = leading_counts.idxmax(axis=1)
    
    # 统计各主导类型的省份数量
    type_summary = leading_counts['主导类型'].value_counts()
    print("各主导类型省份数量：\n", type_summary)
    
    # 输出表格1：各省主导类型及领先年份占比
    # 计算领先年份占比（最高频产业出现次数/总年份数）
    count_cols = leading_counts.columns.drop('主导类型')
    leading_counts['领先占比'] = leading_counts[count_cols].max(axis=1) / len(years)
    table1 = leading_counts[['主导类型', '领先占比']].reset_index()
    table1.columns = ['省份', '主导类型', '领先占比']
    print("\n表1：各省主导类型及领先占比")
    print(table1)
    table1.to_csv(rf"E:\统计建模\统计建模2026\data\表1_各省主导类型.csv", index=False)
    
    # 图1：桑基图展示省份-年份-领先产业的流动（使用sankey）
    # 准备桑基图数据
    sankey_data = df[['省份', '年份', '领先产业']].copy()
    sankey_data['年份'] = sankey_data['年份'].astype(str)
    # 这里简单输出一个热力版的领先产业年度演变矩阵
    pivot_lead = df.pivot_table(index='省份', columns='年份', values='领先产业', aggfunc='first')
    plt.figure(figsize=(14, 10))
    sns.heatmap(pivot_lead.isin(['制造业', '养老产业', '特殊产业']), 
                cmap='tab10', annot=False, cbar=False)
    # 更合适的做法：用颜色映射，这里简化，直接输出表格
    print("\n领先产业年度演变矩阵（部分）：")
    print(pivot_lead.head())
    pivot_lead.to_csv(rf"E:\统计建模\统计建模2026\data\表S1_领先产业年度演变.csv")
    
    return table1, leading_counts

table1, leading_counts = lead_industry_analysis(df_all)

# ================== 2. 省内紧迫度离散程度分析（变异系数） ==================
def dispersion_analysis(df):
    """计算每个省份每年的变异系数CV"""
    industries = sheet_names
    # 计算均值和标准差
    df['均值'] = df[industries].mean(axis=1)
    df['标准差'] = df[industries].std(axis=1)
    # 为避免分母为零，加极小值
    epsilon = 1e-6
    df['CV'] = df['标准差'] / (df['均值'] + epsilon)
    
    # 按省份和年份展示CV
    cv_pivot = df.pivot(index='省份', columns='年份', values='CV')
    print("\n变异系数矩阵（前几行）：")
    print(cv_pivot.head())
    cv_pivot.to_csv(rf"E:\统计建模\统计建模2026\data\表2_变异系数矩阵.csv")
    
    # 分类统计：高离散度(CV>0.5)、中等(0.2<CV<=0.5)、低离散度(CV<=0.2)
    def classify_cv(cv):
        if cv > 0.5:
            return '高离散度'
        elif cv > 0.2:
            return '中等离散度'
        else:
            return '低离散度'
    
    cv_class = cv_pivot.applymap(classify_cv)
    # 多数年份的分类作为省份分类（取众数）
    province_class = cv_class.mode(axis=1)[0]
    class_counts = province_class.value_counts()
    print("\n离散度分类统计：\n", class_counts)
    
    # 图2：热力图
    plt.figure(figsize=(12, 8))
    sns.heatmap(cv_pivot, annot=True, fmt=".2f", cmap='YlOrRd', 
                cbar_kws={'label': '变异系数 CV'})
    plt.title("各省份内部紧迫度变异系数热力图")
    plt.ylabel("省份")
    plt.xlabel("年份")
    plt.tight_layout()
    plt.savefig(rf"E:\统计建模\统计建模2026\fig\图2_变异系数热力图.png", dpi=300)
    plt.show()
    
    # 输出分类统计表
    class_table = pd.DataFrame({
        '省份': province_class.index,
        '离散度类别': province_class.values
    })
    # 合并均值CV
    mean_cv = cv_pivot.mean(axis=1)
    class_table['平均CV'] = mean_cv.values
    class_table.to_csv(rf"E:\统计建模\统计建模2026\data\表2_离散度分类统计.csv", index=False)
    
    return cv_pivot, class_table

cv_pivot, class_table = dispersion_analysis(df_all)

# ================== 3. 基于三分图的省域产业结构聚类 ==================
def ternary_clustering(df, n_clusters=3):
    """三分图坐标转换 + K-means聚类"""
    industries = sheet_names
    # 归一化占比
    total = df[industries].sum(axis=1)
    total = total.replace(0, np.nan)  # 避免除零
    for ind in industries:
        df[f'{ind}_占比'] = df[ind] / total
    df[['制造业_占比', '养老产业_占比', '特殊产业_占比']] = df[industries].div(total, axis=0)
    df.fillna(0, inplace=True)
    
    # 三分图坐标转换 (p1,p2,p3) -> (x,y)
    # 使用标准公式：针对等边三角形，顶点顺序：左下(p2=1), 右下(p1=1), 顶(p3=1)
    # 参考常见转换: x = p2 + p3/2, y = (sqrt(3)/2) * p3
    p1 = df['制造业_占比'].values
    p2 = df['养老产业_占比'].values
    p3 = df['特殊产业_占比'].values
    df['x'] = p2 + p3 / 2
    df['y'] = (np.sqrt(3) / 2) * p3
    
    # K-means 聚类（基于占比向量）
    ratio_data = df[['制造业_占比', '养老产业_占比', '特殊产业_占比']].copy()
    # 标准化
    scaler = StandardScaler()
    ratio_scaled = scaler.fit_transform(ratio_data)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df['聚类标签'] = kmeans.fit_predict(ratio_scaled)
    
    # 聚类中心（原始占比空间）
    centers = scaler.inverse_transform(kmeans.cluster_centers_)
    centers_df = pd.DataFrame(centers, columns=industries)
    centers_df['聚类名称'] = [ '制造聚集型', '养老聚集型', '高危聚集型'][:n_clusters]
    print("\n聚类中心占比：\n", centers_df)
    
    # 绘制三分图
    plt.figure(figsize=(10, 8))
    # 定义三角形顶点坐标
    corners = {
        '制造业': (1, 0),
        '养老产业': (0, 0),
        '特殊产业': (0.5, np.sqrt(3)/2)
    }
    # 绘制三角形边
    tri_x = [corners['制造业'][0], corners['养老产业'][0], corners['特殊产业'][0], corners['制造业'][0]]
    tri_y = [corners['制造业'][1], corners['养老产业'][1], corners['特殊产业'][1], corners['制造业'][1]]
    plt.plot(tri_x, tri_y, 'k-', lw=2)
    # 添加顶点标签
    for label, (cx, cy) in corners.items():
        plt.text(cx, cy, label, fontsize=12, ha='center', va='bottom')
    
    # 散点，按聚类着色
    colors = ['#e41a1c', '#377eb8', '#4daf4a']
    for cluster_id in range(n_clusters):
        subset = df[df['聚类标签'] == cluster_id]
        plt.scatter(subset['x'], subset['y'], c=colors[cluster_id], alpha=0.6, 
                    label=f"聚类{cluster_id+1}: {centers_df.loc[cluster_id, '聚类名称']}")
    plt.xlabel("x 坐标", fontsize=12)
    plt.ylabel("y 坐标", fontsize=12)
    plt.title("各省份产业结构三分图（点代表省份-年份）", fontsize=14)
    plt.legend()
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig(rf"E:\统计建模\统计建模2026\fig\图3_三分图.png", dpi=300)
    plt.show()
    
    # 输出聚类代表性的省份（取每个省份最常见的聚类）
    province_cluster = df.groupby('省份')['聚类标签'].agg(lambda x: x.mode()[0])
    province_cluster_name = province_cluster.map(lambda c: centers_df.loc[c, '聚类名称'])
    cluster_table = pd.DataFrame({
        '省份': province_cluster.index,
        '主导聚类': province_cluster_name.values,
        '平均紧迫度': df.groupby('省份')[industries].mean().mean(axis=1).values
    })
    cluster_table.to_csv(rf"E:\统计建模\统计建模2026\data\表3_产业结构聚类结果.csv", index=False)
    print("\n表3：各省份主导聚类及平均紧迫度")
    print(cluster_table)
    
    return df, centers_df, cluster_table

df_ternary, centers, cluster_table = ternary_clustering(df_all, n_clusters=3)

# ================== 4. 省内耦合协调度分析 ==================
def coupling_coordination(df):
    """计算耦合协调度 D 和总体紧迫度 T"""
    industries = sheet_names
    # 提取 U1,U2,U3
    U1 = df['制造业'].values
    U2 = df['养老产业'].values
    U3 = df['特殊危险环境产业'].values
    # 加极小值避免除零
    eps = 1e-6
    # 耦合度 C
    product = (U1 * U2 * U3) ** (1/3)
    sum_u = U1 + U2 + U3 + eps
    C = 3 * product / sum_u
    # 综合协调指数 T（等权重）
    T = (U1 + U2 + U3) / 3
    # 耦合协调度 D
    D = np.sqrt(C * T)
    df['耦合协调度_D'] = D
    df['总体紧迫度_T'] = T
    
    # 四象限分类（基于2022-2026年均值，或每年）
    # 这里采用平均图，展示2026年（或最后一年的散点）
    last_year = years[-1]
    df_last = df[df['年份'] == last_year].copy()
    
    plt.figure(figsize=(10, 8))
    # 划分象限：使用中位数或均值作为分界
    median_T = df_last['总体紧迫度_T'].median()
    median_D = df_last['耦合协调度_D'].median()
    # 散点
    colors = df_last['聚类标签'].map({0:'red',1:'blue',2:'green'})  # 沿用之前聚类颜色
    plt.scatter(df_last['总体紧迫度_T'], df_last['耦合协调度_D'], c=colors, alpha=0.7, s=80)
    plt.axvline(x=median_T, linestyle='--', color='gray', alpha=0.5)
    plt.axhline(y=median_D, linestyle='--', color='gray', alpha=0.5)
    plt.xlabel("总体紧迫度 T", fontsize=12)
    plt.ylabel("耦合协调度 D", fontsize=12)
    plt.title(f"{last_year} 年各省份耦合协调度四象限图", fontsize=14)
    # 添加省份标签
    for _, row in df_last.iterrows():
        plt.annotate(row['省份'], (row['总体紧迫度_T'], row['耦合协调度_D']), 
                     fontsize=8, alpha=0.7)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(rf"E:\统计建模\统计建模2026\fig\图4_耦合协调四象限_{last_year}.png", dpi=300)
    plt.show()
    
    # 输出分类表格（按四象限）
    def quadrant(t, d, med_t, med_d):
        if t >= med_t and d >= med_d:
            return "高T高D (全面紧迫协调型)"
        elif t >= med_t and d < med_d:
            return "高T低D (单极驱动型)"
        elif t < med_t and d >= med_d:
            return "低T高D (全面均衡滞后型)"
        else:
            return "低T低D (全面滞后型)"
    
    df_last['象限'] = df_last.apply(
        lambda r: quadrant(r['总体紧迫度_T'], r['耦合协调度_D'], median_T, median_D), axis=1)
    quadrant_counts = df_last['象限'].value_counts()
    print("\n四象限分类统计：\n", quadrant_counts)
    
    # 输出典型省份表（每个象限选2-3个）
    table4 = df_last[['省份', '总体紧迫度_T', '耦合协调度_D', '象限']].copy()
    table4.to_csv(rf"E:\统计建模\统计建模2026\data\表4_耦合协调度分类.csv", index=False)
    print("\n表4：各省份耦合协调度及象限分类")
    print(table4)
    
    return df, table4

df_final, table4 = coupling_coordination(df_ternary)

print("\n分析完成！所有图表和表格已保存至当前目录。")