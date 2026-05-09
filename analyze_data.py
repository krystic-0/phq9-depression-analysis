import pandas as pd

# 分析自杀检测数据
print('=== 自杀检测数据 ===')
df = pd.read_csv('outputs/csv/preprocessed_suicide_data.csv')
print('数据形状:', df.shape)
print('class_label分布:')
print(df['class_label'].value_counts())
print('class_label比例:')
print(df['class_label'].value_counts(normalize=True))

# 分析抑郁症状数据
print('\n=== 抑郁症状数据 ===')
df2 = pd.read_csv('outputs/csv/preprocessed_depression_data.csv')
print('数据形状:', df2.shape)
if 'phq9_total' in df2.columns:
    print('PHQ-9总分描述统计:')
    print(df2['phq9_total'].describe())
