import pandas as pd
import numpy as np

print("=" * 60)
print("检查原始数据源")
print("=" * 60)

# 1. 检查抑郁症状数据集
print("\n【数据集1】抑郁症状数据")
print("-" * 60)
try:
    df1 = pd.read_csv("Dataset_14-day_AA_depression_symptoms_mood_and_PHQ-9.csv")
    print(f"数据形状: {df1.shape}")
    print(f"列名: {df1.columns.tolist()}")
    print(f"用户数量: {df1['user_id'].nunique()}")
    print(f"每个用户的平均记录数: {len(df1) / df1['user_id'].nunique():.1f}")
    print(f"\n缺失值统计:")
    print(df1.isnull().sum())
    print(f"\nPHQ-9总分统计:")
    phq_columns = [f"phq{i}" for i in range(1, 10)]
    df1['phq9_total'] = df1[phq_columns].sum(axis=1)
    print(df1['phq9_total'].describe())
    print(f"\n前3行数据:")
    print(df1.head(3))
except Exception as e:
    print(f"读取失败: {e}")

# 2. 检查自杀检测数据集
print("\n\n【数据集2】自杀检测数据")
print("-" * 60)
try:
    df2 = pd.read_csv("Suicide_Detection.csv")
    print(f"数据形状: {df2.shape}")
    print(f"列名: {df2.columns.tolist()}")
    print(f"\n类别分布:")
    print(df2['class'].value_counts())
    print(f"\n文本长度统计:")
    df2['text_length'] = df2['text'].str.len()
    print(df2['text_length'].describe())
    print(f"\n前3行数据:")
    print(df2[['text', 'class']].head(3))
except Exception as e:
    print(f"读取失败: {e}")

# 3. 检查预处理后的数据
print("\n\n【预处理后的数据】")
print("-" * 60)
try:
    df3 = pd.read_csv("outputs/csv/preprocessed_depression_data.csv")
    print(f"抑郁症状数据形状: {df3.shape}")
    print(f"列数: {len(df3.columns)}")
except Exception as e:
    print(f"预处理后抑郁数据读取失败: {e}")

try:
    df4 = pd.read_csv("outputs/csv/preprocessed_suicide_data.csv")
    print(f"自杀检测数据形状: {df4.shape}")
    print(f"列数: {len(df4.columns)}")
    print(f"\n关键特征是否存在:")
    key_features = ['text_length', 'suicide_keyword_count', 'sentiment_score', 
                    'emotion_intensity', 'first_person_ratio', 'negative_word_ratio']
    for feature in key_features:
        exists = feature in df4.columns
        null_count = df4[feature].isnull().sum() if exists else 'N/A'
        print(f"  {feature}: {'✓' if exists else '✗'} (缺失值: {null_count})")
except Exception as e:
    print(f"预处理后自杀数据读取失败: {e}")

print("\n" + "=" * 60)
