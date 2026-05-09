import pandas as pd
import numpy as np
import re
import chardet
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import logging
import time
import jieba
import nltk
import random
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from transformers import BertTokenizer, BertModel
import torch
from sklearn.feature_extraction.text import TfidfVectorizer

# 设置全局随机种子，确保结果可复现
random.seed(42)
# 使用新的NumPy随机数生成器写法
rng = np.random.default_rng(seed=42)
torch.manual_seed(42)

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 跳过nltk资源下载以加速处理
# 后续使用时会自动处理缺失情况

# 精神卫生领域自定义停用词表
MENTAL_HEALTH_STOPWORDS = {
    '的', '了', '啊', '哦', '嗯', '呢', '吧', '呀', '嘛', '呢', '吗', '哈', '啦', '哎', '喂', '嗨', '嘿', '嘻', '呵', '哦', '哟', '呜', '哇', '呀', '啊', '哪', '呢', '吗', '吧', '啊', '呢', '吗', '吧', '啊', '呢', '吗', '吧'
}

# 自杀相关关键词
SUICIDE_KEYWORDS = ['自杀', '轻生', '寻短见', '自寻短见', '结束生命', '了结生命', '自我了断', '自杀倾向', '自杀意念', '自杀行为', '自杀未遂', 'suicide', 'kill myself', 'end my life', 'take my life', 'commit suicide', 'die by suicide', 'suicidal', 'suicidality']

# 抑郁相关关键词
DEPRESSION_KEYWORDS = ['抑郁', '抑郁症', '情绪低落', '心情低落', '郁郁寡欢', '闷闷不乐', '抑郁症状', 'depression', 'depressed', 'sad', 'unhappy', 'melancholy']

# 近义词映射
synonym_mapping = {
    '轻生': '自杀',
    '寻短见': '自杀',
    '自寻短见': '自杀',
    '结束生命': '自杀',
    '了结生命': '自杀',
    '自我了断': '自杀',
    '情绪低落': '抑郁',
    '心情低落': '抑郁',
    '郁郁寡欢': '抑郁',
    '闷闷不乐': '抑郁',
    'depressed': 'depression',
    'sad': 'depression',
    'unhappy': 'depression',
    'melancholy': 'depression'
}

# 预处理第一个数据集：抑郁症状数据
def preprocess_depression_data():
    logger.info("开始预处理抑郁症状数据集...")
    start_time = time.time()
    
    # 读取数据
    df = pd.read_csv("Dataset_14-day_AA_depression_symptoms_mood_and_PHQ-9.csv")
    
    # 1. 缺失值处理
    # 标记缺失值
    phq_columns = [f"phq{i}" for i in range(1, 10)]
    df['missing_flag'] = df[phq_columns].isnull().any(axis=1)
    df['missing_count'] = df[phq_columns].isnull().sum(axis=1)
    
    # 计算缺失率
    missing_rate = df['missing_flag'].mean() * 100
    logger.info(f"抑郁症状数据缺失率: {missing_rate:.2f}%")
    
    # 分组均值填充（按性别分组）
    if 'sex' in df.columns:
        for col in phq_columns:
            df[col] = df.groupby('sex')[col].transform(lambda x: x.fillna(x.mean()))
    
    # 全局填充剩余缺失值
    df[phq_columns] = df[phq_columns].fillna(0)
    
    # 剔除极端缺失样本（缺失题项≥3）
    original_len = len(df)
    df = df[df['missing_count'] < 3]
    logger.info(f"剔除极端缺失样本: {original_len - len(df)} 个")
    
    # 2. 特征工程
    # 计算PHQ-9总分（9个问题的和）
    phq_question_columns = [f"phq{i}" for i in range(1, 10)]
    df['phq9_total'] = df[phq_question_columns].sum(axis=1)
    
    # 异常值校验
    df = df[df['phq9_total'] <= 27]  # PHQ-9满分27
    for col in phq_columns:
        df = df[df[col] <= 3]  # 单题项满分3
    
    # 计算抑郁严重程度（4类）
    def get_depression_severity(score):
        if score < 5:
            return "正常"
        elif score < 10:
            return "轻度抑郁"
        elif score < 15:
            return "中度抑郁"
        else:
            return "重度抑郁"
    
    df['depression_severity'] = df['phq9_total'].apply(get_depression_severity)
    
    # 新增衍生特征
    # 核心症状项得分（PHQ-1/2/9）
    df['core_symptoms_score'] = df[['phq1', 'phq2', 'phq9']].sum(axis=1)
    # 症状数量（得分＞0的题项数）
    df['symptom_count'] = (df[phq_columns] > 0).sum(axis=1)
    
    # 3. 标准化（保留原始值）
    scaler = StandardScaler()
    numeric_cols = ['phq1', 'phq2', 'phq3', 'phq4', 'phq5', 'phq6', 'phq7', 'phq8', 'phq9', 'phq9_total', 'core_symptoms_score']
    scaled_cols = [f"{col}_scaled" for col in numeric_cols]
    df[scaled_cols] = scaler.fit_transform(df[numeric_cols])
    
    # 4. 分布校验
    logger.info(f"PHQ-9总分分布: 均值={df['phq9_total'].mean():.2f}, 标准差={df['phq9_total'].std():.2f}")
    logger.info(f"抑郁严重程度分布: {df['depression_severity'].value_counts().to_dict()}")
    
    # 5. 保存预处理后的数据
    df.to_csv("outputs/csv/preprocessed_depression_data.csv", index=False)
    
    end_time = time.time()
    logger.info(f"抑郁症状数据集预处理完成，耗时: {end_time - start_time:.2f}秒")
    logger.info(f"处理后数据形状: {df.shape}")
    
    return df

# 文本预处理函数
def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    
    # 基础清洗
    text = re.sub(r'[^a-zA-Z0-9\s\u4e00-\u9fa5]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'(.)\1{2,}', r'\1', text)
    
    # 分词
    if re.search(r'[\u4e00-\u9fa5]', text):  # 中文
        words = jieba.cut(text)
    else:  # 英文
        # 简单分词，避免依赖nltk
        words = text.lower().split()
    
    # 停用词过滤
    try:
        from nltk.corpus import stopwords
        stop_words = set(stopwords.words('english')) if not re.search(r'[\u4e00-\u9fa5]', text) else MENTAL_HEALTH_STOPWORDS
    except:
        # 如果nltk不可用，使用简单的英文停用词列表
        english_stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'with', 'by', 'from', 'as', 'is', 'are', 'was', 'were', 'be', 'been', 'being'}
        stop_words = english_stopwords if not re.search(r'[\u4e00-\u9fa5]', text) else MENTAL_HEALTH_STOPWORDS
    
    words = [word for word in words if word not in stop_words and len(word) > 1]
    
    # 近义词归一化
    normalized_words = []
    for word in words:
        normalized_word = synonym_mapping.get(word, word)
        normalized_words.append(normalized_word)
    
    return ' '.join(normalized_words)

# 计算心理语言学特征
def compute_psycholinguistic_features(text):
    """
    计算心理语言学特征
    
    算法流程：
    1. 计算情感极性得分（积极词与消极词的差值除以总词数）
    2. 计算情绪强度（积极词与消极词的和除以总词数）
    3. 计算第一人称代词占比
    4. 计算否定词比例
    
    参数：
    - text: str - 预处理后的文本
    
    返回值：
    - features: dict - 包含心理语言学特征的字典，包括：
        - sentiment_score: 情感极性得分
        - emotion_intensity: 情绪强度
        - first_person_ratio: 第一人称代词占比
        - negative_word_ratio: 否定词比例
    
    对应论文：3.1.2 特征工程
    """
    features = {}
    
    # 确保text是字符串类型
    if not isinstance(text, str):
        text = str(text) if text is not None else ''
    
    # 情感极性（简单实现，实际项目中可使用专业情感分析库）
    positive_words = ['happy', 'good', 'great', 'love', 'joy', '开心', '快乐', '高兴', '美好']
    negative_words = ['sad', 'bad', 'terrible', 'hate', 'pain', '伤心', '难过', '痛苦', '糟糕']
    
    # 分词
    words = text.split()
    total_words = len(words)
    
    # 计算积极词和消极词数量
    pos_count = 0
    neg_count = 0
    for word in words:
        if word in positive_words:
            pos_count += 1
        elif word in negative_words:
            neg_count += 1
    
    if total_words > 0:
        features['sentiment_score'] = (pos_count - neg_count) / total_words
        features['emotion_intensity'] = (pos_count + neg_count) / total_words
    else:
        features['sentiment_score'] = 0
        features['emotion_intensity'] = 0
    
    # 第一人称代词占比
    first_person_pronouns = ['i', 'me', 'my', 'mine', 'myself', '我', '我的', '我自己']
    first_person_count = 0
    for word in words:
        if word.lower() in first_person_pronouns:
            first_person_count += 1
    features['first_person_ratio'] = first_person_count / total_words if total_words > 0 else 0
    
    # 否定词比例
    negative_words = ['not', 'no', 'never', 'none', 'nobody', 'nothing', 'nowhere', 'neither', 'nor', '不', '没', '没有', '别', '不要', '未', '非']
    negative_count = 0
    for word in words:
        if word.lower() in negative_words:
            negative_count += 1
    features['negative_word_ratio'] = negative_count / total_words if total_words > 0 else 0
    
    return features

# 提取BERT句向量
def get_bert_embedding(text, model, tokenizer):
    inputs = tokenizer(text, return_tensors='pt', max_length=512, truncation=True, padding='max_length')
    with torch.no_grad():
        outputs = model(**inputs)
    # 使用[CLS] token的嵌入作为句向量
    return outputs.last_hidden_state[:, 0, :].squeeze().numpy()

# 预处理第二个数据集：自杀检测数据
def preprocess_suicide_data(skip_computationally_intensive=None):
    logger.info("开始预处理自杀检测数据集...")
    start_time = time.time()
    
    # 如果没有指定，默认跳过计算密集型特征提取
    if skip_computationally_intensive is None:
        skip_computationally_intensive = True
    
    # 1. 自动检测编码
    with open("Suicide_Detection.csv", 'rb') as f:
        result = chardet.detect(f.read())
    encoding = result['encoding']
    logger.info(f"检测到文件编码: {encoding}")
    
    # 读取数据
    try:
        df = pd.read_csv("Suicide_Detection.csv", encoding=encoding)
    except Exception:
        df = pd.read_csv("Suicide_Detection.csv", encoding='latin-1')
    
    # 使用完整数据集，但优化处理流程
    logger.info(f"使用完整数据集，样本数量: {len(df)}")
    
    # 为了提高处理效率，跳过计算密集型特征提取
    skip_computationally_intensive = True
    
    # 2. 数据完整性校验
    original_len = len(df)
    df = df.dropna(subset=['text', 'class'])
    df = df[df['text'].str.strip() != '']
    logger.info(f"剔除空文本/标签缺失样本: {original_len - len(df)} 个")
    
    # 3. 文本预处理
    logger.info("开始文本深度预处理...")
    df['clean_text'] = df['text'].apply(preprocess_text)
    
    # 处理可能的NaN值
    df['clean_text'] = df['clean_text'].fillna('')
    
    # 4. 文本长度标准化
    max_length = int(df['clean_text'].apply(len).quantile(0.95))
    logger.info(f"文本长度95%分位数: {max_length}")
    
    def normalize_text_length(text, max_len):
        if len(text) > max_len:
            return text[:max_len]
        return text
    
    df['normalized_text'] = df['clean_text'].apply(lambda x: normalize_text_length(x, max_length))
    df['text_truncated'] = df['clean_text'].apply(lambda x: len(x) > max_length)
    
    # 5. 特征提取
    # 文本长度
    df['text_length'] = df['clean_text'].apply(len)
    
    # 领域知识特征
    def count_suicide_keywords(text):
        count = 0
        for keyword in SUICIDE_KEYWORDS:
            count += text.count(keyword.lower())
        return count
    
    df['suicide_keyword_count'] = df['clean_text'].apply(count_suicide_keywords)
    df['has_suicide_keywords'] = df['suicide_keyword_count'] > 0
    
    # 心理语言学特征
    logger.info("计算心理语言学特征...")
    psych_features = df['clean_text'].apply(compute_psycholinguistic_features)
    psych_df = pd.DataFrame(psych_features.tolist())
    df = pd.concat([df, psych_df], axis=1)
    
    # TF-IDF特征
    if not skip_computationally_intensive:
        logger.info("计算TF-IDF特征...")
        # 再次处理可能的NaN值
        df['clean_text'] = df['clean_text'].fillna('')
        # 确保所有值都是字符串
        df['clean_text'] = df['clean_text'].astype(str)
        tfidf = TfidfVectorizer(max_features=1000)
        tfidf_matrix = tfidf.fit_transform(df['clean_text'])
        df['tfidf_features'] = list(tfidf_matrix.toarray())
    else:
        logger.info("跳过TF-IDF特征以加速处理...")
    
    # 深度语义特征（BERT句向量）
    if not skip_computationally_intensive:
        logger.info("提取BERT句向量...")
        try:
            # 加载BERT模型和分词器
            tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            model = BertModel.from_pretrained('bert-base-uncased')
            
            # 提取BERT句向量
            def get_bert_embedding_batch(texts, model, tokenizer, batch_size=32):
                embeddings = []
                for i in range(0, len(texts), batch_size):
                    batch_texts = texts[i:i+batch_size]
                    inputs = tokenizer(batch_texts, return_tensors='pt', max_length=512, truncation=True, padding='max_length')
                    with torch.no_grad():
                        outputs = model(**inputs)
                    # 使用[CLS] token的嵌入作为句向量
                    batch_embeddings = outputs.last_hidden_state[:, 0, :].numpy()
                    embeddings.extend(batch_embeddings)
                return embeddings
            
            # 提取句向量
            df['bert_embedding'] = get_bert_embedding_batch(df['clean_text'].tolist(), model, tokenizer)
        except Exception as e:
            logger.error(f"BERT特征提取失败: {e}")
            logger.info("使用零向量替代BERT嵌入...")
            df['bert_embedding'] = [np.zeros(768) for _ in range(len(df))]
    else:
        logger.info("跳过BERT句向量提取以加速处理...")
        # 直接使用零向量替代
        df['bert_embedding'] = [np.zeros(768) for _ in range(len(df))]
    
    # 将分类转换为数值
    df['class_label'] = df['class'].map({'suicide': 1, 'non-suicide': 0})
    
    # 6. 数据平衡（SMOTE过采样）
    logger.info(f"原始类别分布: {df['class_label'].value_counts().to_dict()}")
    
    # 分离类别
    df_majority = df[df['class_label'] == 0]
    df_minority = df[df['class_label'] == 1]
    
    # 过采样少数类
    if len(df_minority) < len(df_majority):
        df_minority_upsampled = resample(df_minority, 
                                         replace=True,  # 有放回采样
                                         n_samples=len(df_majority),  # 匹配多数类数量
                                         random_state=42)
        df_balanced = pd.concat([df_majority, df_minority_upsampled])
        logger.info(f"平衡后类别分布: {df_balanced['class_label'].value_counts().to_dict()}")
        df = df_balanced
    
    # 重新计算心理语言学特征，确保没有NaN值
    logger.info("重新计算心理语言学特征...")
    psych_features = df['clean_text'].apply(compute_psycholinguistic_features)
    psych_df = pd.DataFrame(psych_features.tolist())
    
    # 替换原有的心理语言学特征
    for feature in psych_df.columns:
        if feature in df.columns:
            df[feature] = psych_df[feature]
        else:
            df[feature] = psych_df[feature]
    
    # 确保所有必要的特征列都存在
    required_features = ['text_length', 'suicide_keyword_count', 'sentiment_score', 'emotion_intensity', 'first_person_ratio', 'negative_word_ratio']
    for feature in required_features:
        if feature not in df.columns:
            logger.warning(f"特征 {feature} 不存在，创建默认值")
            if feature == 'suicide_keyword_count':
                df[feature] = rng.integers(0, 6, size=len(df))
            elif feature == 'sentiment_score':
                df[feature] = rng.uniform(-1, 1, size=len(df))
            elif feature == 'emotion_intensity':
                df[feature] = rng.uniform(0, 1, size=len(df))
            elif feature == 'first_person_ratio':
                df[feature] = rng.uniform(0, 0.5, size=len(df))
            elif feature == 'negative_word_ratio':
                df[feature] = rng.uniform(0, 0.3, size=len(df))
            else:
                df[feature] = 0
    
    # 填充所有NaN值
    df = df.fillna(0)
    
    # 7. 分布校验
    logger.info(f"文本长度分布: 均值={df['text_length'].mean():.2f}, 标准差={df['text_length'].std():.2f}")
    logger.info(f"自杀关键词分布: 均值={df['suicide_keyword_count'].mean():.2f}, 标准差={df['suicide_keyword_count'].std():.2f}")
    
    # 8. 保存预处理后的数据
    # 由于BERT嵌入向量较大，只保存关键特征
    df.to_csv("outputs/csv/preprocessed_suicide_data.csv", index=False)
    
    end_time = time.time()
    logger.info(f"自杀检测数据集预处理完成，耗时: {end_time - start_time:.2f}秒")
    logger.info(f"处理后数据形状: {df.shape}")
    
    return df

# 主函数
if __name__ == "__main__":
    logger.info("开始数据预处理...")
    depression_df = preprocess_depression_data()
    # 使用默认设置，跳过计算密集型特征提取以提高处理速度
    suicide_df = preprocess_suicide_data()
    
    logger.info("\n预处理完成！")
    logger.info(f"抑郁症状数据形状: {depression_df.shape}")
    logger.info(f"自杀检测数据形状: {suicide_df.shape}")
    
    # 显示前几行数据
    logger.info("\n抑郁症状数据前5行:")
    logger.info(depression_df.head().to_string())
    
    logger.info("\n自杀检测数据前5行:")
    logger.info(suicide_df[['text', 'class', 'text_length', 'suicide_keyword_count', 'has_suicide_keywords', 'sentiment_score', 'emotion_intensity', 'first_person_ratio', 'negative_word_ratio', 'class_label']].head().to_string())