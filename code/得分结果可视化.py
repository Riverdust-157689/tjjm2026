import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
ALPHA = 0.6
BETA = 0.4
prefix=f"A{ALPHA}B{BETA}"

matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

#xl = pd.ExcelFile(rf'E:\统计建模\统计建模2026\data\综合紧迫度结果_修正{prefix}.xlsx')
xl = pd.ExcelFile(rf'E:\统计建模\统计建模2026\data\综合紧迫度结果.xlsx')
#sheets = xl.sheet_names[3:6]
sheets = xl.sheet_names[0:3]
print(sheets)
fig, axes = plt.subplots(1, 3, figsize=(20, 12), constrained_layout=True)

for ax, sheet in zip(axes, sheets):
    df = pd.read_excel(xl, sheet_name=sheet, header=0)
    first_col = df.columns[0]
    df = df.set_index(first_col)
    df.columns = df.columns.astype(int)

    data = df.values.astype(float)
    provinces = df.index.tolist()
    years = df.columns.tolist()

    im = ax.imshow(data, aspect='auto', cmap='Reds')
    ax.set_xticks(range(len(years)))
    ax.set_xticklabels(years, rotation=45)
    ax.set_yticks(range(len(provinces)))
    ax.set_yticklabels(provinces, fontsize=7)
    ax.set_title(sheet, fontsize=14)
    ax.set_xlabel('年份')
    ax.set_ylabel('省份')
    plt.colorbar(im, ax=ax, label='紧迫度')

fig.suptitle('各省各年三产业紧迫度热力图', fontsize=16)
#plt.savefig(r'E:\统计建模\统计建模2026\fig\紧迫度热力图.png', dpi=150, bbox_inches='tight')
plt.savefig(rf'E:\统计建模\统计建模2026\fig\紧迫度热力图_修正{prefix}.png', dpi=150, bbox_inches='tight')
plt.show()