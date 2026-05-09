import pandas as pd
import numpy as np

print("=" * 60)
print("检查原始数据")
print("=" * 60)

# 直接读取原始数据集
df = pd.read_csv("Dataset_14-day_AA_depression_symptoms_mood_and_PHQ-9.csv")
print(f"原始数据形状: {df.shape}")
print(f"列名: {df.columns.tolist()}")

# 检查前10行数据
print("\n前10行原始数据:")
print(df.head(10))

# 检查PHQ列
phq_columns = [f"phq{i}" for i in range(1, 10)]
print("\nPHQ列数据类型:")
for col in phq_columns:
    if col in df.columns:
        print(f"{col}: {df[col].dtype}")

# 计算PHQ-9总分
df['phq9_total'] = df[phq_columns].sum(axis=1)
print("\nPHQ-9总分分布:")
print(df['phq9_total'].describe())

# 按用户分组，检查PHQ-9总分是否变化
user_groups = df.groupby('user_id')
print("\n每个用户的PHQ-9总分情况:")
for user_id, group in user_groups:
    if len(group) > 1:
        scores = group['phq9_total'].unique()
        if len(scores) == 1:
            print(f"用户 {user_id}: PHQ-9总分始终为 {scores[0]} (记录数: {len(group)})")
        else:
            print(f"用户 {user_id}: PHQ-9总分有变化: {sorted(scores)}")

# 检查phq.day列
print("\nphq.day列分析:")
print(f"phq.day最小值: {df['phq.day'].min()}")
print(f"phq.day最大值: {df['phq.day'].max()}")
print(f"phq.day唯一值数量: {df['phq.day'].nunique()}")

# 检查时间相关列
print("\n时间相关列:")
if 'time' in df.columns:
    print(f"time列存在，前5个值: {df['time'].head().tolist()}")
if 'period.name' in df.columns:
    print(f"period.name列存在，唯一值: {df['period.name'].unique()}")
if 'start.time' in df.columns:
    print(f"start.time列存在，前5个值: {df['start.time'].head().tolist()}")

# 检查其他重要列
print("\n其他重要列:")
if 'happiness.score' in df.columns:
    print(f"happiness.score分布: {df['happiness.score'].describe()}")
if 'age' in df.columns:
    print(f"age分布: {df['age'].describe()}")
if 'sex' in df.columns:
    print(f"sex分布: {df['sex'].value_counts()}")

print("\n" + "=" * 60)
