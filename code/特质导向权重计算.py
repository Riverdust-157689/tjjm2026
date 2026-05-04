# -*- coding: utf-8 -*-
"""
统计建模比赛 - 三大产业具身智能引入紧迫性 得分与权重计算
功能：负向指标处理 + 加权得分计算 + 归一化最终权重
"""
import pandas as pd

# ===================== 1. 录入原始数据（直接对应你提供的表格）=====================
# 评价指标
indicators = [
    "岗位危险性", "场景非结构化水平", "任务非标准化水平",
    "操作精细要求", "安全替代价值", "体力消耗程度",
    "环境适应性要求", "社交/情感交互需求", "工作枯燥/重复性"
]
# 影响方向：+正向（越高越紧迫），-负向（越高越不紧迫）
direction = ["+", "+", "+", "-", "+", "+", "+", "-", "+"]
# 各产业原始评分
score_manufacture = [5, 4, 3, 8, 6, 6, 4, 1, 8]    # 制造业
score_elderly = [3, 8, 9, 5, 4, 8, 2, 9, 4]         # 养老产业
score_danger = [9, 9, 8, 7, 10, 9, 10, 5, 2]        # 特殊产业
# 各指标权重
weight_manufacture = [0.08, 0.10, 0.08, 0.18, 0.12, 0.10, 0.05, 0.02, 0.27]
weight_elderly = [0.05, 0.15, 0.15, 0.08, 0.05, 0.12, 0.02, 0.23, 0.15]
weight_danger = [0.18, 0.12, 0.10, 0.08, 0.20, 0.12, 0.15, 0.02, 0.03]

# 构建数据框（方便查看和计算）
df = pd.DataFrame({
    "评价指标": indicators,
    "影响方向": direction,
    "制造业评分": score_manufacture,
    "养老产业评分": score_elderly,
    "特殊产业评分": score_danger,
    "制造业权重": weight_manufacture,
    "养老产业权重": weight_elderly,
    "特殊产业权重": weight_danger
})

# ===================== 2. 负向指标处理（核心：反向计分，满分10分）=====================
# 负向指标公式：处理后得分 = 10 - 原始得分
df["制造业_处理后得分"] = df.apply(lambda x: 10-x["制造业评分"] if x["影响方向"]=="-" else x["制造业评分"], axis=1)
df["养老产业_处理后得分"] = df.apply(lambda x: 10-x["养老产业评分"] if x["影响方向"]=="-" else x["养老产业评分"], axis=1)
df["特殊产业_处理后得分"] = df.apply(lambda x: 10-x["特殊产业评分"] if x["影响方向"]=="-" else x["特殊产业评分"], axis=1)

# ===================== 3. 计算单指标加权得分 =====================
df["制造业_加权得分"] = df["制造业_处理后得分"] * df["制造业权重"]
df["养老产业_加权得分"] = df["养老产业_处理后得分"] * df["养老产业权重"]
df["特殊产业_加权得分"] = df["特殊产业_处理后得分"] * df["特殊产业权重"]

# ===================== 4. 计算产业总紧迫性得分 =====================
total_manufacture = df["制造业_加权得分"].sum()
total_elderly = df["养老产业_加权得分"].sum()
total_danger = df["特殊产业_加权得分"].sum()

# ===================== 5. 归一化处理 → 最终三大产业权重（和为1）=====================
total_sum = total_manufacture + total_elderly + total_danger
final_weight_manufacture = total_manufacture / total_sum
final_weight_elderly = total_elderly / total_sum
final_weight_danger = total_danger / total_sum

# ===================== 6. 输出结果（格式化展示，可直接复制到论文）=====================
print("="*80)
print("【三大产业引入具身智能 指标处理与加权计算明细】")
print("="*80)
detail_cols = ["评价指标", "影响方向", "制造业_处理后得分", "养老产业_处理后得分", "特殊产业_处理后得分",
               "制造业_加权得分", "养老产业_加权得分", "特殊产业_加权得分"]
print(df[detail_cols].round(4).to_string(index=False))

print("\n" + "="*80)
print("【三大产业行业紧迫性总得分】")
print("="*80)
print(f"制造业总得分：{total_manufacture:.4f}")
print(f"养老产业总得分：{total_elderly:.4f}")
print(f"特殊产业总得分：{total_danger:.4f}")

print("\n" + "="*80)
print("【归一化后最终权重（用于省内产业需求度评价，和为1）】")
print("="*80)
print(f"制造业权重：{final_weight_manufacture:.4f}")
print(f"养老产业权重：{final_weight_elderly:.4f}")
print(f"特殊产业权重：{final_weight_danger:.4f}")
print(f"权重校验（求和）：{final_weight_manufacture+final_weight_elderly+final_weight_danger:.4f}")