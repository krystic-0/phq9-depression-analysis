import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import time
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class OptimizedClustering:
    def __init__(self, n_clusters=3, random_state=42):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.scaler = StandardScaler()
    
    def preprocess_features(self, X):
        """预处理特征：标准化"""
        logger.info("开始特征预处理...")
        X_scaled = self.scaler.fit_transform(X)
        return X_scaled
    
    def cluster_data(self, X):
        """聚类数据"""
        logger.info("开始聚类...")
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=self.random_state, n_init=10)
        labels = kmeans.fit_predict(X)
        return labels, kmeans
    
    def reduce_dimension_tsne(self, X, perplexity=3, max_iter=1000, learning_rate=100):
        """使用t-SNE降维"""
        logger.info(f"开始t-SNE降维（perplexity={perplexity}, max_iter={max_iter}, learning_rate={learning_rate}）...")
        tsne = TSNE(
            n_components=2,
            random_state=self.random_state,
            perplexity=perplexity,
            learning_rate=learning_rate,
            n_jobs=-1  # 使用所有CPU核心
        )
        tsne_result = tsne.fit_transform(X)
        return tsne_result
    
    def reduce_dimension_pca(self, X):
        """使用PCA降维"""
        logger.info("开始PCA降维...")
        pca = PCA(n_components=2, random_state=self.random_state)
        pca_result = pca.fit_transform(X)
        return pca_result
    
    def visualize_clusters(self, X_2d, labels, title="聚类可视化"):
        """可视化聚类结果"""
        logger.info("开始可视化...")
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=labels, cmap='viridis', s=100)
        plt.colorbar(scatter, label='簇')
        plt.title(title)
        plt.xlabel('维度1')
        plt.ylabel('维度2')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'outputs/images/{title.replace(" ", "_").lower()}.png')
        plt.close()
        logger.info(f"可视化结果已保存为 outputs/images/{title.replace(" ", "_").lower()}.png")
    
    def run(self, X):
        """运行完整的聚类和可视化流程"""
        start_time = time.time()
        
        # 1. 预处理特征
        X_scaled = self.preprocess_features(X)
        
        # 2. 聚类
        labels, kmeans = self.cluster_data(X_scaled)
        
        # 3. 降维（t-SNE）
        n_samples = X.shape[0]
        # 对于小样本，perplexity应该设置得更小
        perplexity = min(10, n_samples - 1)
        tsne_result = self.reduce_dimension_tsne(X_scaled, perplexity=perplexity)
        
        # 4. 降维（PCA，作为对比）
        pca_result = self.reduce_dimension_pca(X_scaled)
        
        # 5. 可视化
        self.visualize_clusters(tsne_result, labels, "t_SNE聚类可视化")
        self.visualize_clusters(pca_result, labels, "PCA聚类可视化")
        
        end_time = time.time()
        logger.info(f"聚类和可视化完成，耗时: {end_time - start_time:.2f}秒")
        
        return labels, tsne_result, pca_result

if __name__ == "__main__":
    # 示例：使用抑郁症状数据进行聚类
    logger.info("开始优化聚类分析...")
    
    # 加载预处理后的数据
    try:
        depression_df = pd.read_csv("outputs/csv/preprocessed_depression_data.csv")
        
        # 选择特征
        features = [f"phq{i}_scaled" for i in range(1, 10)] + ['core_symptoms_score_scaled', 'symptom_count']
        X = depression_df[features].dropna()
        
        logger.info(f"数据形状: {X.shape}")
        
        # 运行优化的聚类分析
        clustering = OptimizedClustering(n_clusters=3, random_state=42)
        labels, tsne_result, pca_result = clustering.run(X)
        
        # 保存聚类结果
        depression_df.loc[X.index, 'optimized_cluster'] = labels
        depression_df.to_csv("outputs/csv/preprocessed_depression_data.csv", index=False)
        logger.info("聚类结果已保存")
        
    except Exception as e:
        logger.error(f"运行失败: {e}")
        # 创建模拟数据进行演示
        logger.info("创建模拟数据进行演示...")
        # 生成12个样本，10维特征的模拟数据
        np.random.seed(42)
        X = np.random.randn(12, 10)
        # 添加一些结构，使聚类更明显
        X[:4, :5] += 2  # 簇0
        X[4:8, 5:] += 2  # 簇1
        X[8:, :] -= 2     # 簇2
        
        logger.info(f"模拟数据形状: {X.shape}")
        
        # 运行优化的聚类分析
        clustering = OptimizedClustering(n_clusters=3, random_state=42)
        labels, tsne_result, pca_result = clustering.run(X)
        
        logger.info(f"模拟数据聚类标签: {labels}")
        logger.info("演示完成")
