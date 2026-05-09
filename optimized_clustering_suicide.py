import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import plotly.graph_objects as go
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
    
    def reduce_dimension_tsne(self, X, perplexity=30, learning_rate=200):
        """使用t-SNE降维"""
        logger.info(f"开始t-SNE降维（perplexity={perplexity}, learning_rate={learning_rate}）...")
        # 对于大数据集，使用样本进行降维
        if len(X) > 500:
            logger.info(f"样本数量较大（{len(X)}），使用500个样本进行t-SNE降维...")
            np.random.seed(self.random_state)
            sample_idx = np.random.choice(len(X), 500, replace=False)
            X_sample = X[sample_idx]
        else:
            X_sample = X
        
        tsne = TSNE(
            n_components=2, 
            random_state=self.random_state, 
            perplexity=perplexity, 
            learning_rate=learning_rate,
            n_jobs=-1  # 使用所有CPU核心
        )
        tsne_result = tsne.fit_transform(X_sample)
        return tsne_result, sample_idx if len(X) > 500 else np.arange(len(X))
    
    def visualize_clusters(self, X_2d, labels, title="聚类可视化"):
        """可视化聚类结果"""
        logger.info("开始可视化...")
        
        try:
            # 创建DataFrame
            df = pd.DataFrame({
                "tsne_1": X_2d[:, 0],
                "tsne_2": X_2d[:, 1],
                "cluster": labels
            })
            
            # 淡紫色渐变配色
            purple_colors = ['#4B0082', '#6A0DAD', '#9370DB', '#F5F0FF']
            
            # 使用Plotly Express创建散点图
            fig = px.scatter(
                df,
                x="tsne_1",
                y="tsne_2",
                color="cluster",
                title="t-SNE 聚类可视化分析",
                labels={"tsne_1": "t-SNE 1", "tsne_2": "t-SNE 2", "cluster": "簇编号"},
                color_continuous_scale=purple_colors,
                opacity=0.8,
                template="plotly_white"
            )
            
            # 优化布局
            fig.update_layout(
                title=dict(
                    text="t-SNE 聚类可视化分析",
                    x=0.5,  # 居中
                    font=dict(
                        size=16,
                        color="#333333"  # 深灰色
                    )
                ),
                xaxis=dict(
                    title="t-SNE 1",
                    gridcolor="#F0F0F0"
                ),
                yaxis=dict(
                    title="t-SNE 2",
                    gridcolor="#F0F0F0"
                ),
                coloraxis_colorbar=dict(
                    title="簇编号"
                )
            )
            
            # 优化标记大小
            fig.update_traces(
                marker=dict(
                    size=5,  # 大小适中
                    line=dict(
                        width=0.5,
                        color="white"
                    )
                )
            )
            
            # 只保存为PNG图片，不生成HTML
            png_path = f'outputs/images/{title.replace(" ", "_").lower()}.png'
            fig.write_image(png_path, scale=2)
            
            logger.info(f"可视化结果已保存为 {png_path}")
        except Exception as e:
            logger.error(f"可视化失败: {e}")
            # 创建一个简单的错误图片
            import matplotlib.pyplot as plt
            plt.figure(figsize=(10, 6))
            plt.title("聚类可视化")
            plt.text(0.5, 0.5, f"可视化失败: {str(e)}", ha='center', va='center')
            plt.axis('off')
            png_path = f'outputs/images/{title.replace(" ", "_").lower()}.png'
            plt.savefig(png_path, dpi=150)
            plt.close()
            logger.info(f"错误图片已保存为 {png_path}")
    
    def run(self, X):
        """运行完整的聚类和可视化流程"""
        start_time = time.time()
        
        # 1. 预处理特征
        X_scaled = self.preprocess_features(X)
        
        # 2. 聚类
        labels, kmeans = self.cluster_data(X_scaled)
        
        # 3. 降维（t-SNE）- 进一步减少计算量
        n_samples = X.shape[0]
        # 对于不同规模的样本，设置不同的perplexity
        if n_samples < 50:
            perplexity = min(10, n_samples - 1)
        elif n_samples < 200:
            perplexity = 20
        else:
            perplexity = 30
        
        # 进一步减少样本数量
        tsne_result, sample_idx = self.reduce_dimension_tsne(X_scaled, perplexity=perplexity, learning_rate=200)
        
        # 4. 可视化 - 只生成t-SNE图
        try:
            self.visualize_clusters(tsne_result, labels[sample_idx], "t_SNE自杀检测聚类可视化")
        except Exception as e:
            logger.warning(f"可视化失败，但聚类结果已生成: {e}")

        end_time = time.time()
        logger.info(f"聚类和可视化完成，耗时: {end_time - start_time:.2f}秒")

        return labels, tsne_result

if __name__ == "__main__":
    # 使用自杀检测数据进行聚类
    logger.info("开始优化聚类分析（自杀检测数据集）...")
    
    # 加载预处理后的数据
    try:
        # 限制数据量，只使用前10000条记录
        logger.info("加载并限制数据量...")
        suicide_df = pd.read_csv("outputs/csv/preprocessed_suicide_data.csv", nrows=10000)
        
        # 直接使用基本特征，避免解析TF-IDF特征的开销
        logger.info("使用基本特征进行聚类...")
        features = ['text_length', 'suicide_keyword_count', 'sentiment_score', 'emotion_intensity']
        X = suicide_df[features].dropna().values
        
        # 进一步限制样本数量
        if len(X) > 5000:
            logger.info(f"样本数量较大（{len(X)}），使用5000个样本进行聚类...")
            np.random.seed(42)
            sample_idx = np.random.choice(len(X), 5000, replace=False)
            X = X[sample_idx]
        
        logger.info(f"数据形状: {X.shape}")
        
        # 运行优化的聚类分析
        clustering = OptimizedClustering(n_clusters=3, random_state=42)
        labels, tsne_result = clustering.run(X)
        
        # 保存聚类结果
        logger.info("保存聚类结果...")
        # 由于我们使用了样本数据，只保存对应样本的聚类结果
        for i, cluster in enumerate(labels):
            if i < len(suicide_df):
                suicide_df.loc[i, 'optimized_cluster'] = cluster
        
        # 只保存前10000条记录，避免修改完整数据集
        suicide_df.to_csv("outputs/csv/clustering_results_sample.csv", index=False)
        logger.info("聚类结果已保存到 outputs/csv/clustering_results_sample.csv")
        
        # 分析聚类结果
        cluster_counts = pd.Series(labels).value_counts()
        logger.info(f"聚类分布: {cluster_counts.to_dict()}")
        
    except Exception as e:
        logger.error(f"运行失败: {e}")
        logger.info("请确保自杀检测数据集已正确预处理")
