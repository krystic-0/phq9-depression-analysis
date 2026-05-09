import pandas as pd
import numpy as np

print("=" * 60)
print("正确分析PHQ数据结构")
print("=" * 60)

# 读取原始数据
df = pd.read_csv("Dataset_14-day_AA_depression_symptoms_mood_and_PHQ-9.csv")
print(f"数据形状: {df.shape}")

# 提取PHQ相关列
phq_columns = []
for col in df.columns:
    if col.startswith('phq') or col.startswith('Q'):
        phq_columns.append(col)

print(f"\nPHQ相关列: {phq_columns}")
print(f"PHQ相关列数量: {len(phq_columns)}")

# 分析前5个用户的PHQ分数变化
user_ids = df['user_id'].unique()[:5]
print("\n前5个用户的PHQ分数变化:")

for user_id in user_ids:
    user_data = df[df['user_id'] == user_id].iloc[0]  # 每个用户只取一条记录（包含所有天数的分数）
    
    # 提取PHQ分数
    phq_scores = []
    for col in phq_columns:
        if pd.notna(user_data[col]):
            phq_scores.append(user_data[col])
    
    if phq_scores:
        scores_unique = np.unique(phq_scores)
        print(f"\n用户 {user_id}:")
        print(f"PHQ分数数量: {len(phq_scores)}")
        print(f"唯一分数数量: {len(scores_unique)}")
        print(f"分数范围: {min(phq_scores):.1f} - {max(phq_scores):.1f}")
        print(f"分数变化: {'有变化' if len(scores_unique) > 1 else '无变化'}")
        
        # 显示前10个分数
        print("前10个分数:")
        for i, score in enumerate(phq_scores[:10]):
            print(f"第{i+1}天: {score:.1f}")

# 分析整体数据
print("\n" + "=" * 60)
print("整体数据分析:")

# 计算每个用户的PHQ分数变化
user_change_counts = []
for user_id in df['user_id'].unique():
    user_data = df[df['user_id'] == user_id].iloc[0]
    phq_scores = []
    for col in phq_columns:
        if pd.notna(user_data[col]):
            phq_scores.append(user_data[col])
    if phq_scores:
        unique_scores = np.unique(phq_scores)
        user_change_counts.append(len(unique_scores))

if user_change_counts:
    print(f"用户数量: {len(user_change_counts)}")
    print(f"平均唯一分数数量: {np.mean(user_change_counts):.2f}")
    print(f"有变化的用户比例: {(np.array(user_change_counts) > 1).mean() * 100:.2f}%")

print("\n" + "=" * 60)
