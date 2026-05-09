import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from visualization import plot_tsne_clustering
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 加载预处理后的数据
depression_df = pd.read_csv("outputs/csv/preprocessed_depression_data.csv")

# 选择特征
features = [f"phq{i}_scaled" for i in range(1, 10)] + ['core_symptoms_score_scaled', 'symptom_count']
X = depression_df[features].dropna()

# 限制样本数量以提高速度
if len(X) > 1000:
    X_sample = X.sample(1000, random_state=42)
else:
    X_sample = X

# 选择K=3进行聚类
best_k = 3
logger.info(f"选择最佳簇数: {best_k}")

# K-Means聚类
kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=5)
kmeans_labels = kmeans.fit_predict(X)

# 保存聚类结果
depression_df.loc[X.index, 'kmeans_cluster'] = kmeans_labels

# 降维可视化（使用样本数据）
logger.info("进行降维可视化...")
# 根据样本数量动态调整perplexity参数
n_samples = len(X_sample)
perplexity = min(30, n_samples - 1)
logger.info(f"使用perplexity={perplexity}（样本数量={n_samples}）")
# 使用max_iter替代n_iter以避免警告
tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity, max_iter=300)
tsne_result = tsne.fit_transform(X_sample)

# 生成图表并保存
fig = plot_tsne_clustering(tsne_result, kmeans_labels[:len(tsne_result)])
fig.write_image('outputs/images/clustering_visualization.png')
logger.info("聚类可视化图已保存为 outputs/images/clustering_visualization.png")

logger.info("聚类分析完成！")