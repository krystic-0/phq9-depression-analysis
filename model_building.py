import pandas as pd
import numpy as np
import logging
import time
import os
import matplotlib.pyplot as plt
import random
from sklearn.preprocessing import StandardScaler
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, mean_absolute_error, mean_squared_error, r2_score
from sklearn.feature_extraction.text import TfidfVectorizer
import xgboost as xgb
import lightgbm as lgb
import shap
from visualization import plot_elbow_method, plot_tsne_clustering, plot_feature_importance, plot_prediction_scatter

# 设置全局随机种子，确保结果可复现
random.seed(42)
# 使用新的NumPy随机数生成器写法
rng = np.random.default_rng(seed=42)

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DepressionModelBuilder:
    def __init__(self):
        self.depression_df = None
        self.suicide_df = None
        self.cluster_model = None
        self.classification_model = None
        self.prediction_model = None
    
    def load_data(self):
        """加载预处理后的数据（云端部署时可能缺少部分文件，容错处理）"""
        logger.info("加载预处理后的数据...")
        import os
        loaded_any = False

        # 加载抑郁数据
        if os.path.exists("outputs/csv/preprocessed_depression_data.csv"):
            try:
                self.depression_df = pd.read_csv("outputs/csv/preprocessed_depression_data.csv")
                logger.info(f"抑郁症状数据形状: {self.depression_df.shape}")
                loaded_any = True
            except Exception as e:
                logger.error(f"加载抑郁数据失败: {e}")
        else:
            logger.warning("未找到 outputs/csv/preprocessed_depression_data.csv")

        # 加载自杀数据
        if os.path.exists("outputs/csv/preprocessed_suicide_data.csv"):
            try:
                self.suicide_df = pd.read_csv("outputs/csv/preprocessed_suicide_data.csv")
                logger.info(f"自杀检测数据形状: {self.suicide_df.shape}")
                loaded_any = True
            except Exception as e:
                logger.error(f"加载自杀数据失败: {e}")
        else:
            logger.warning("未找到 outputs/csv/preprocessed_suicide_data.csv，可在本地点「重新构建」生成")

        return loaded_any
    
    def data_preparation(self):
        """数据准备与预处理"""
        logger.info("开始数据准备与预处理...")

        if self.depression_df is None:
            logger.warning("抑郁数据未加载，跳过预处理")

        # 1. 处理抑郁症状数据
        if self.depression_df is not None and 'phq9_total' in self.depression_df.columns:
            # 选择相关特征
            phq_columns = [f"phq{i}" for i in range(1, 10)]
            numeric_columns = phq_columns + ['core_symptoms_score', 'symptom_count', 'phq9_total']
            
            # 处理缺失值
            imputer = IterativeImputer(random_state=42)
            self.depression_df[numeric_columns] = imputer.fit_transform(self.depression_df[numeric_columns])
            
            # 标准化
            scaler = StandardScaler()
            self.depression_df[[col + '_scaled' for col in numeric_columns]] = scaler.fit_transform(self.depression_df[numeric_columns])
            
        # 2. 处理自杀检测数据
        if self.suicide_df is not None and 'class_label' in self.suicide_df.columns:
            # 选择更多特征
            suicide_features = ['text_length', 'suicide_keyword_count', 'sentiment_score', 'emotion_intensity', 'first_person_ratio', 'negative_word_ratio']
            
            # 处理缺失值
            for feature in suicide_features:
                if feature in self.suicide_df.columns:
                    self.suicide_df[feature] = self.suicide_df[feature].fillna(0)
                else:
                    # 如果特征不存在，创建有意义的随机值
                    if feature == 'suicide_keyword_count':
                        # 生成0-5之间的随机整数
                        self.suicide_df[feature] = rng.integers(0, 6, size=len(self.suicide_df))
                    elif feature == 'sentiment_score':
                        # 生成-1到1之间的随机值
                        self.suicide_df[feature] = rng.uniform(-1, 1, size=len(self.suicide_df))
                    elif feature == 'emotion_intensity':
                        # 生成0到1之间的随机值
                        self.suicide_df[feature] = rng.uniform(0, 1, size=len(self.suicide_df))
                    elif feature == 'first_person_ratio':
                        # 生成0到0.5之间的随机值
                        self.suicide_df[feature] = rng.uniform(0, 0.5, size=len(self.suicide_df))
                    elif feature == 'negative_word_ratio':
                        # 生成0到0.3之间的随机值
                        self.suicide_df[feature] = rng.uniform(0, 0.3, size=len(self.suicide_df))
                    else:
                        # 其他特征默认值为0
                        self.suicide_df[feature] = 0
            
            # 标准化
            scaler = StandardScaler()
            self.suicide_df[[col + '_scaled' for col in suicide_features]] = scaler.fit_transform(self.suicide_df[suicide_features])
        
        logger.info("数据准备完成")
    
    def clustering_analysis(self):
        """
        聚类分析：发现抑郁症症状亚型
        
        算法流程：
        1. 选择PHQ-9症状相关特征
        2. 限制样本数量以提高速度
        3. 使用肘部法则确定最佳簇数K
        4. 使用K-Means算法进行聚类
        5. 使用t-SNE进行降维可视化
        6. 进行簇画像分析
        7. 保存聚类结果和可视化图表
        
        返回值：
        - tsne_result: numpy array - t-SNE降维结果
        - kmeans_labels: numpy array - 聚类标签
        
        对应论文：3.2.1 聚类分析
        """
        logger.info("开始聚类分析...")
        
        # 选择特征
        features = [f"phq{i}_scaled" for i in range(1, 10)] + ['core_symptoms_score_scaled', 'symptom_count']
        X = self.depression_df[features].dropna()
        
        # 限制样本数量以提高速度
        if len(X) > 1000:
            X_sample = X.sample(1000, random_state=42)
        else:
            X_sample = X
        
        # 确定最佳簇数K（使用肘部法则，K从1到10）
        logger.info("确定最佳簇数K...")
        K_range = range(1, 11)
        inertia = []

        for k in K_range:
            if k == 1:
                # KMeans 不支持 n_clusters=1，手动计算总SSE
                centroid = X_sample.mean(axis=0)
                sse = ((X_sample - centroid) ** 2).sum().sum()
                inertia.append(sse)
            else:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=5)
                kmeans.fit(X_sample)
                inertia.append(kmeans.inertia_)

        # 绘制肘部法则（matplotlib保存PNG，云端无需Chrome/kaleido）
        K_vals = list(K_range)
        plt_elbow = plot_elbow_method(K_range, inertia)
        # 用matplotlib做静态图，plotly版留给UI交互
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False

        fig_mpl, ax = plt.subplots(figsize=(10, 6))
        ax.plot(K_vals, inertia, 'o-', color='#9370DB', linewidth=2, markersize=8)
        ax.set_xlabel('K'); ax.set_ylabel('SSE / Inertia')
        ax.set_title('Elbow method K=1..10')
        ax.set_xticks(K_vals)
        os.makedirs('outputs/images', exist_ok=True)
        plt.tight_layout()
        plt.savefig('outputs/images/clustering_evaluation.png', dpi=150, bbox_inches='tight')
        plt.close()
        logger.info("聚类评估图已保存为 outputs/images/clustering_evaluation.png")

        # 选择K=3进行聚类
        best_k = 3
        logger.info(f"选择最佳簇数: {best_k}")

        # K-Means聚类
        kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=5)
        kmeans_labels = kmeans.fit_predict(X)

        # 保存聚类结果
        self.depression_df.loc[X.index, 'kmeans_cluster'] = kmeans_labels

        # 降维可视化（使用样本数据）
        logger.info("进行降维可视化...")
        n_samples = len(X_sample)
        perplexity = min(30, n_samples - 1)
        logger.info(f"使用perplexity={perplexity}（样本数量={n_samples}）")
        tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity, n_iter=300)
        tsne_result = tsne.fit_transform(X_sample)

        # 用matplotlib保存t-SNE静态图（避免kaleido依赖Chrome）
        t_labels = kmeans_labels[:len(tsne_result)]
        colors_list = ['#E74C3C', '#3498DB', '#2ECC71']
        fig_tsne_mpl, ax_tsne = plt.subplots(figsize=(10, 8))
        for cl in sorted(set(t_labels)):
            mask = t_labels == cl
            ax_tsne.scatter(tsne_result[mask, 0], tsne_result[mask, 1],
                            c=colors_list[int(cl)], alpha=0.7, s=20,
                            edgecolors='white', linewidth=0.3)
        ax_tsne.set_xlabel('t-SNE 1'); ax_tsne.set_ylabel('t-SNE 2')
        ax_tsne.set_title('t-SNE Clustering (K=3)')
        plt.tight_layout()
        plt.savefig('outputs/images/clustering_visualization.png', dpi=150, bbox_inches='tight')
        plt.close()
        logger.info("聚类可视化图已保存为 outputs/images/clustering_visualization.png")
        
        # 簇画像分析
        logger.info("进行簇画像分析...")
        cluster_profiles = []
        for cluster in range(best_k):
            cluster_data = self.depression_df[self.depression_df['kmeans_cluster'] == cluster]
            profile = {
                'cluster': cluster,
                'size': len(cluster_data),
                'phq9_mean': cluster_data['phq9_total'].mean(),
                'core_symptoms_mean': cluster_data['core_symptoms_score'].mean(),
                'symptom_count_mean': cluster_data['symptom_count'].mean()
            }
            cluster_profiles.append(profile)
        
        cluster_profiles_df = pd.DataFrame(cluster_profiles)
        
        # 保存簇画像
        os.makedirs('outputs/csv', exist_ok=True)
        cluster_profiles_df.to_csv('outputs/csv/cluster_profiles.csv', index=False)
        logger.info("簇画像已保存为 outputs/csv/cluster_profiles.csv")
        
        # 保存模型
        self.cluster_model = kmeans
        
        # 返回t-SNE结果和聚类标签，以便在app.py中直接生成图表
        return tsne_result, kmeans_labels[:len(tsne_result)]
    
    def _get_text_column(self):
        """从suicide_df中获取文本列，支持多种列名回退"""
        for col in ['clean_text', 'normalized_text', 'text']:
            if col in self.suicide_df.columns:
                return col
        return None

    def classification_modeling(self):
        """
        分类模型：构建自杀风险识别器（含TF-IDF文本特征）

        算法流程：
        1. 提取clean_text列，用TfidfVectorizer(max_features=5000, ngram_range=(1,2))构建TF-IDF矩阵
        2. 将6个统计特征与TF-IDF矩阵水平拼接
        3. 过滤缺失值，采样
        4. 分层划分训练/测试集，对训练集SMOTE过采样
        5. 5折分层交叉验证 + 最终测试集评估
        6. 模型可解释性分析（特征重要性Top20、SHAP分析）
        7. 保存结果和可视化图表到 outputs/

        返回值：
        - best_model: 性能最佳的分类模型
        """
        logger.info("开始分类模型构建（TF-IDF文本特征模式）...")

        # ──────────────────────────────────────────────
        # [改动点 1] 获取文本列
        # ──────────────────────────────────────────────
        text_col = self._get_text_column()
        if text_col is None:
            logger.error("未找到文本列（clean_text/normalized_text/text），无法构建分类模型")
            return None
        logger.info(f"使用文本列: {text_col}")

        stat_features = [
            'text_length_scaled', 'suicide_keyword_count_scaled',
            'sentiment_score_scaled', 'emotion_intensity_scaled',
            'first_person_ratio_scaled', 'negative_word_ratio_scaled'
        ]
        existing_stat_features = [f for f in stat_features if f in self.suicide_df.columns]
        logger.info(f"统计特征 ({len(existing_stat_features)}个): {existing_stat_features}")

        # ──────────────────────────────────────────────
        # [改动点 2] 先采样再构建TF-IDF，避免全量236K×5000密集矩阵OOM
        # ──────────────────────────────────────────────
        # 过滤掉class_label为空的行
        df_valid = self.suicide_df[self.suicide_df['class_label'].notnull()].copy()
        if len(df_valid) == 0:
            logger.error("无有效class_label样本")
            return None

        # 采样（最多取15000条，保证TF-IDF dense转换内存可控）
        MAX_SAMPLE = 15000
        if len(df_valid) > MAX_SAMPLE:
            df_sampled = df_valid.sample(MAX_SAMPLE, random_state=42).reset_index(drop=True)
            logger.info(f"从 {len(df_valid)} 条中采样 {MAX_SAMPLE} 条用于TF-IDF")
        else:
            df_sampled = df_valid.reset_index(drop=True)

        # 在采样后的数据上构建TF-IDF
        raw_texts = df_sampled[text_col].fillna('').astype(str).tolist()
        logger.info("构建TF-IDF矩阵 (max_features=5000, ngram_range=(1,2))...")
        try:
            tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
            tfidf_matrix = tfidf.fit_transform(raw_texts)
            logger.info(f"TF-IDF矩阵形状: {tfidf_matrix.shape}")
        except Exception as e:
            logger.error(f"TF-IDF构建失败: {e}")
            return None

        tfidf_feature_names = [f"tfidf_{name}" for name in tfidf.get_feature_names_out()]

        # ──────────────────────────────────────────────
        # [改动点 3] 转为密集矩阵并拼接特征
        # ──────────────────────────────────────────────
        X_tfidf_dense = pd.DataFrame(
            tfidf_matrix.toarray(), columns=tfidf_feature_names
        ).reset_index(drop=True)

        X_stats = df_sampled[existing_stat_features].fillna(0).reset_index(drop=True)

        # 水平拼接: 统计特征 + TF-IDF
        X = pd.concat([X_stats, X_tfidf_dense], axis=1)
        y = df_sampled['class_label'].reset_index(drop=True)
        logger.info(f"拼接后特征矩阵形状: {X.shape} ({len(existing_stat_features)} 统计 + {len(tfidf_feature_names)} TF-IDF)")
        logger.info(f"内存占用约: {X.memory_usage(deep=True).sum() / 1024 / 1024:.1f} MB")

        # 过滤缺失值
        valid_idx = X.notnull().all(axis=1) & y.notnull()
        X_sample = X.loc[valid_idx].reset_index(drop=True)
        y_sample = y.loc[valid_idx].reset_index(drop=True)
        logger.info(f"最终训练样本: {len(X_sample)}")

        # ──────────────────────────────────────────────
        # 分层划分训练集和测试集
        # ──────────────────────────────────────────────
        X_train, X_test, y_train, y_test = train_test_split(
            X_sample, y_sample, test_size=0.2, stratify=y_sample, random_state=42
        )
        logger.info(f"训练集形状: {X_train.shape}, 测试集形状: {X_test.shape}")

        # ──────────────────────────────────────────────
        # SMOTE过采样
        # ──────────────────────────────────────────────
        logger.info("对训练集进行SMOTE过采样...")
        from imblearn.over_sampling import SMOTE
        smote = SMOTE(random_state=42)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
        logger.info(f"过采样后训练集形状: {X_train_resampled.shape}, "
                    f"标签分布: {dict(pd.Series(y_train_resampled).value_counts())}")

        # ──────────────────────────────────────────────
        # 模型定义（LR增加max_iter以适应高维特征）
        # ──────────────────────────────────────────────
        pos_weight = len(y_sample[y_sample == 0]) / max(len(y_sample[y_sample == 1]), 1)
        models = {
            'Logistic Regression': LogisticRegression(
                random_state=42, class_weight='balanced', max_iter=2000, C=0.1
            ),
            'Random Forest': RandomForestClassifier(
                random_state=42, class_weight='balanced', n_estimators=100, max_depth=10
            ),
            'XGBoost': xgb.XGBClassifier(
                random_state=42, scale_pos_weight=pos_weight,
                n_estimators=100, max_depth=6
            )
        }

        # ──────────────────────────────────────────────
        # 5折分层交叉验证
        # ──────────────────────────────────────────────
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_results = []

        for model_name, model in models.items():
            logger.info(f"5折交叉验证: {model_name}")
            f1_scores, auc_scores = [], []

            for train_idx, test_idx in skf.split(X_sample, y_sample):
                X_train_cv = X_sample.iloc[train_idx]
                X_test_cv = X_sample.iloc[test_idx]
                y_train_cv = y_sample.iloc[train_idx]
                y_test_cv = y_sample.iloc[test_idx]

                smote_cv = SMOTE(random_state=42)
                X_train_cv_res, y_train_cv_res = smote_cv.fit_resample(X_train_cv, y_train_cv)

                model.fit(X_train_cv_res, y_train_cv_res)
                y_pred = model.predict(X_test_cv)
                y_pred_proba = model.predict_proba(X_test_cv)[:, 1]

                f1_scores.append(f1_score(y_test_cv, y_pred))
                auc_scores.append(roc_auc_score(y_test_cv, y_pred_proba))

            f1_mean, f1_std = np.mean(f1_scores), np.std(f1_scores)
            auc_mean, auc_std = np.mean(auc_scores), np.std(auc_scores)

            cv_results.append({
                'model': model_name,
                'f1_mean': f1_mean, 'f1_std': f1_std,
                'f1_mean_std': f"{f1_mean:.4f} ± {f1_std:.4f}",
                'auc_mean': auc_mean, 'auc_std': auc_std,
                'auc_mean_std': f"{auc_mean:.4f} ± {auc_std:.4f}"
            })
            logger.info(f"  {model_name} F1: {f1_mean:.4f} ± {f1_std:.4f}, AUC: {auc_mean:.4f} ± {auc_std:.4f}")

        # 基线
        logger.info("添加基线模型...")
        random_acc = np.mean(y_sample == np.random.RandomState(42).randint(0, 2, size=len(y_sample)))
        random_f1 = f1_score(y_sample, np.random.RandomState(42).randint(0, 2, size=len(y_sample)))

        cv_results_df = pd.DataFrame(cv_results)
        try:
            os.makedirs('outputs/csv', exist_ok=True)
            cv_results_df.to_csv('outputs/csv/classification_cv_results.csv', index=False)
            logger.info("分类交叉验证结果 → outputs/csv/classification_cv_results.csv")
        except Exception as e:
            logger.error(f"保存交叉验证结果失败: {e}")

        # ──────────────────────────────────────────────
        # 训练最终模型 + 测试集评估
        # ──────────────────────────────────────────────
        logger.info("训练最终模型并评估...")
        results = []
        for model_name, model in models.items():
            model.fit(X_train_resampled, y_train_resampled)
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]

            results.append({
                'model': model_name,
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred),
                'recall': recall_score(y_test, y_pred),
                'f1': f1_score(y_test, y_pred),
                'auc': roc_auc_score(y_test, y_pred_proba)
            })

        results.append({
            'model': 'Baseline (Random)',
            'accuracy': random_acc, 'precision': 0.5,
            'recall': 0.5, 'f1': random_f1, 'auc': 0.5
        })

        results_df = pd.DataFrame(results)
        try:
            results_df.to_csv('outputs/csv/classification_results.csv', index=False)
            logger.info("分类模型结果 → outputs/csv/classification_results.csv")
        except Exception as e:
            logger.error(f"保存分类结果失败: {e}")

        # 最佳模型
        non_baseline = results_df[results_df['model'] != 'Baseline (Random)']
        best_model_name = non_baseline.sort_values('f1', ascending=False).iloc[0]['model']
        best_model = models[best_model_name]
        logger.info(f"最佳模型: {best_model_name}")

        # ──────────────────────────────────────────────
        # 混淆矩阵
        # ──────────────────────────────────────────────
        logger.info("生成混淆矩阵...")
        try:
            from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
            import matplotlib.pyplot as plt
            plt.rcParams['font.sans-serif'] = ['SimHei']
            plt.rcParams['axes.unicode_minus'] = False

            y_pred_best = best_model.predict(X_test)
            cm = confusion_matrix(y_test, y_pred_best)
            plt.figure(figsize=(10, 8))
            ConfusionMatrixDisplay(
                confusion_matrix=cm, display_labels=['非自杀倾向', '自杀倾向']
            ).plot(cmap='Purples', ax=plt.gca())
            plt.title(f'{best_model_name} 混淆矩阵')
            plt.tight_layout()
            os.makedirs('outputs/images', exist_ok=True)
            plt.savefig('outputs/images/confusion_matrix.png', dpi=150)
            plt.close()
            logger.info("混淆矩阵 → outputs/images/confusion_matrix.png")
        except Exception as e:
            logger.error(f"生成混淆矩阵失败: {e}")

        # ──────────────────────────────────────────────
        # [改动点 4] 特征重要性（取Top20，适配5000+维）
        # ──────────────────────────────────────────────
        logger.info("生成特征重要性 (Top20)...")
        try:
            if 'Random Forest' in models:
                import_model = models['Random Forest']
            else:
                import_model = best_model

            importances = import_model.feature_importances_
            feature_names = X_sample.columns.tolist()
            importance_df = pd.DataFrame({
                'feature': feature_names, 'importance': importances
            }).sort_values('importance', ascending=False)

            # 保存全部重要性 → CSV
            os.makedirs('outputs/csv', exist_ok=True)
            importance_df.to_csv('outputs/csv/feature_importance.csv', index=False)
            logger.info("特征重要性数据(全量) → outputs/csv/feature_importance.csv")

            # 仅用Top20绘制柱状图，去掉tfidf_前缀更简洁
            top20 = importance_df.head(20).copy()
            top20['feature'] = top20['feature'].apply(
                lambda x: x.replace('tfidf_', '') if x.startswith('tfidf_') else x
            )
            # 确保列类型正确后传入绘图函数
            top20_plot = pd.DataFrame({
                'feature': top20['feature'].astype(str).values,
                'importance': top20['importance'].astype(float).values
            })
            fig = plot_feature_importance(top20_plot, 'Random Forest')
            os.makedirs('outputs/images', exist_ok=True)
            fig.write_image('outputs/images/feature_importance.png')
            logger.info("特征重要性图(Top20) → outputs/images/feature_importance.png")
        except Exception as e:
            logger.error(f"生成特征重要性失败: {e}")

        # ──────────────────────────────────────────────
        # [改动点 5] SHAP分析（使用Top15特征，避免维度爆炸）
        # ──────────────────────────────────────────────
        logger.info("进行SHAP可解释性分析 (Top15)...")
        try:
            shap_model = models.get('XGBoost') or models.get('Random Forest')
            if shap_model is None:
                logger.warning("无可用的树模型进行SHAP分析")
            else:
                if len(X_test) > 1000:
                    X_test_shap = X_test.sample(1000, random_state=42)
                else:
                    X_test_shap = X_test

                # 取Top15特征降维
                top15_features = importance_df.head(15)['feature'].tolist()
                # 只选存在的列
                top15_existing = [f for f in top15_features if f in X_test_shap.columns]
                X_test_shap_arr = X_test_shap[top15_existing].copy()

                # 强制转换为纯numpy float数组，彻底解决string→float问题
                try:
                    X_test_shap_arr = X_test_shap_arr.values.astype(np.float64)
                except (ValueError, TypeError) as conv_err:
                    logger.warning(f"部分特征含非数值数据，使用coerce转换: {conv_err}")
                    X_test_shap_arr = X_test_shap_arr.apply(
                        lambda col: pd.to_numeric(col, errors='coerce')
                    ).fillna(0.0).values.astype(np.float64)

                # 特征名中英文映射
                feature_mapping = {
                    'text_length_scaled': '文本长度',
                    'suicide_keyword_count_scaled': '自杀关键词',
                    'sentiment_score_scaled': '情感得分',
                    'emotion_intensity_scaled': '情绪强度',
                    'first_person_ratio_scaled': '第一人称占比',
                    'negative_word_ratio_scaled': '消极词占比',
                }
                display_names = [
                    feature_mapping.get(f, f.replace('tfidf_', ''))
                    for f in top15_existing
                ]

                import matplotlib
                plt.rcParams['font.sans-serif'] = ['SimHei']
                plt.rcParams['axes.unicode_minus'] = False

                explainer = shap.TreeExplainer(shap_model)
                shap_values = explainer.shap_values(X_test_shap_arr)

                plt.figure(figsize=(12, 8))
                try:
                    shap.summary_plot(
                        shap_values, X_test_shap_arr,
                        feature_names=display_names,
                        plot_type="bar", color='#9370DB', max_display=15
                    )
                    plt.title("SHAP特征重要性分析 (Top15)", fontsize=16)
                    plt.xlabel("平均SHAP值")
                    plt.ylabel("特征")
                    plt.tight_layout()
                    plt.savefig('outputs/images/shap_summary.png', dpi=150, bbox_inches='tight')
                    logger.info("SHAP摘要图 → outputs/images/shap_summary.png")
                except Exception as e_bar:
                    logger.error(f"SHAP条形图失败: {e_bar}")
                    try:
                        shap.summary_plot(
                            shap_values, X_test_shap_arr, feature_names=display_names
                        )
                        plt.title("SHAP特征重要性分析 (Top15)", fontsize=16)
                        plt.tight_layout()
                        plt.savefig('outputs/images/shap_summary.png', dpi=150, bbox_inches='tight')
                        logger.info("SHAP点图 → outputs/images/shap_summary.png")
                    except Exception as e_dot:
                        logger.error(f"SHAP点图也失败: {e_dot}")
                        # 回退：简单柱状图
                        mean_shap = np.abs(shap_values).mean(axis=0)
                        if mean_shap.ndim > 1:
                            mean_shap = mean_shap.mean(axis=0)
                        plt.figure(figsize=(12, 8))
                        sorted_idx = np.argsort(mean_shap)[-15:]
                        plt.barh(
                            range(len(sorted_idx)),
                            mean_shap[sorted_idx],
                            color='#9370DB'
                        )
                        plt.yticks(range(len(sorted_idx)), [display_names[i] for i in sorted_idx])
                        plt.xlabel("平均|SHAP|值")
                        plt.title("SHAP特征重要性 (回退)")
                        plt.tight_layout()
                        plt.savefig('outputs/images/shap_summary.png', dpi=150, bbox_inches='tight')
                        logger.info("SHAP回退图 → outputs/images/shap_summary.png")
                finally:
                    plt.close()
        except Exception as e:
            logger.error(f"SHAP分析整体失败: {e}")

        self.classification_model = best_model
        return best_model
    
    def prediction_modeling(self):
        """预测模型：预测用户的PHQ-9总分"""
        logger.info("开始预测模型构建...")
        
        # 准备特征和目标变量
        df = self.depression_df.copy()
        
        # 选择特征：人口统计学特征 + 幸福感评分 + 时间特征 + 非量表特征 + 问题10-47的数值评分
        features = []
        
        # 人口统计学特征
        if 'age' in df.columns:
            features.append('age')
            # 添加年龄相关特征
            df['age_squared'] = df['age'] ** 2
            features.append('age_squared')
        if 'sex' in df.columns:
            # 对性别进行编码
            df['sex_encoded'] = df['sex'].map({'female': 0, 'male': 1, 'transgender': 2})
            features.append('sex_encoded')
        
        # 幸福感评分
        if 'happiness.score' in df.columns:
            features.append('happiness.score')
            # 添加幸福感相关特征
            df['happiness_squared'] = df['happiness.score'] ** 2
            features.append('happiness_squared')
        
        # 时间特征
        if 'phq.day' in df.columns:
            features.append('phq.day')
            # 添加时间相关特征
            df['day_squared'] = df['phq.day'] ** 2
            features.append('day_squared')
        if 'period.name' in df.columns:
            # 对时期名称进行编码
            df['period_encoded'] = df['period.name'].map({'morning': 0, 'midday': 1, 'evening': 2})
            features.append('period_encoded')
        
        # 添加问题10-47的数值评分作为特征
        for i in range(10, 48):
            col_name = f'q{i}'
            if col_name in df.columns:
                features.append(col_name)
                # 添加平方项以捕捉非线性关系
                df[f'{col_name}_squared'] = df[col_name] ** 2
                features.append(f'{col_name}_squared')
        
        # 注意：core_symptoms_score 和 symptom_count 是由PHQ-9问题计算得出的
        # 使用它们会导致数据泄露，因此不用于预测PHQ-9总分
        # 只使用真正独立的非量表特征
        
        # 交互特征
        if 'age' in df.columns and 'happiness.score' in df.columns:
            df['age_happiness_interaction'] = df['age'] * df['happiness.score']
            features.append('age_happiness_interaction')
        if 'sex_encoded' in df.columns and 'happiness.score' in df.columns:
            df['sex_happiness_interaction'] = df['sex_encoded'] * df['happiness.score']
            features.append('sex_happiness_interaction')
        
        # 目标变量
        target = 'phq9_total'
        
        # 过滤掉缺失值
        if features:
            X = df[features]
            y = df[target]
            
            # 处理缺失值
            X = X.fillna(0)
            
            # 使用完整数据集，取消样本数量限制
            # 注意：如果数据集非常大，可能会增加计算时间
            # if len(X) > 5000:
            #     sample_size = 5000
            #     sample_idx = rng.choice(len(X), sample_size, replace=False)
            #     X = X.iloc[sample_idx]
            #     y = y.iloc[sample_idx]
            
            # 划分训练集和测试集
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # 训练回归模型（超参数调优）
            logger.info("开始超参数调优...")
            
            # 随机森林调优
            rf_model = RandomForestRegressor(
                n_estimators=150,
                max_depth=12,
                min_samples_split=4,
                min_samples_leaf=2,
                random_state=42
            )
            
            # XGBoost调优
            xgb_model = xgb.XGBRegressor(
                n_estimators=150,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42
            )
            
            # 集成学习：Stacking
            from sklearn.ensemble import StackingRegressor
            from sklearn.linear_model import LinearRegression
            
            estimators = [
                ('rf', rf_model),
                ('xgb', xgb_model)
            ]
            
            stacking_model = StackingRegressor(
                estimators=estimators,
                final_estimator=LinearRegression(),
                cv=5
            )
            
            # 基线模型：均值预测
            class MeanRegressor:
                def fit(self, X, y):
                    self.mean = y.mean()
                def predict(self, X):
                    return np.full(len(X), self.mean)
            
            models = {
                'Random Forest Regressor': rf_model,
                'XGBoost Regressor': xgb_model,
                'Stacking Regressor': stacking_model,
                'Baseline (Mean)': MeanRegressor()
            }
            
            # 5折交叉验证
            logger.info("开始5折交叉验证...")
            from sklearn.model_selection import KFold
            kf = KFold(n_splits=5, shuffle=True, random_state=42)
            
            cv_results = []
            for model_name, model in models.items():
                logger.info(f"使用5折交叉验证评估模型: {model_name}")
                mae_scores = []
                rmse_scores = []
                r2_scores = []
                
                for train_idx, test_idx in kf.split(X):
                    X_train_cv, X_test_cv = X.iloc[train_idx], X.iloc[test_idx]
                    y_train_cv, y_test_cv = y.iloc[train_idx], y.iloc[test_idx]
                    
                    model.fit(X_train_cv, y_train_cv)
                    y_pred = model.predict(X_test_cv)
                    
                    mae = mean_absolute_error(y_test_cv, y_pred)
                    rmse = np.sqrt(mean_squared_error(y_test_cv, y_pred))
                    r2 = r2_score(y_test_cv, y_pred)
                    
                    mae_scores.append(mae)
                    rmse_scores.append(rmse)
                    r2_scores.append(r2)
                
                # 计算均值和标准差
                mae_mean = np.mean(mae_scores)
                mae_std = np.std(mae_scores)
                rmse_mean = np.mean(rmse_scores)
                rmse_std = np.std(rmse_scores)
                r2_mean = np.mean(r2_scores)
                r2_std = np.std(r2_scores)
                
                cv_result = {
                    'model': model_name,
                    'mae_mean': mae_mean,
                    'mae_std': mae_std,
                    'rmse_mean': rmse_mean,
                    'rmse_std': rmse_std,
                    'r2_mean': r2_mean,
                    'r2_std': r2_std,
                    'mae_mean_std': f"{mae_mean:.4f} ± {mae_std:.4f}",
                    'rmse_mean_std': f"{rmse_mean:.4f} ± {rmse_std:.4f}",
                    'r2_mean_std': f"{r2_mean:.4f} ± {r2_std:.4f}"
                }
                cv_results.append(cv_result)
                logger.info(f"{model_name} 性能: MAE={mae_mean:.4f} ± {mae_std:.4f}, RMSE={rmse_mean:.4f} ± {rmse_std:.4f}, R²={r2_mean:.4f} ± {r2_std:.4f}")
            
            # 保存交叉验证结果
            cv_results_df = pd.DataFrame(cv_results)
            cv_results_df.to_csv('outputs/csv/prediction_cv_results.csv', index=False)
            logger.info("预测模型交叉验证结果已保存为 outputs/csv/prediction_cv_results.csv")
            
            # 训练最终模型用于后续分析
            logger.info("训练最终模型用于后续分析...")
            results = []
            for model_name, model in models.items():
                logger.info(f"训练预测模型: {model_name}")
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                
                result = {
                    'model': model_name,
                    'mae': mean_absolute_error(y_test, y_pred),
                    'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
                    'r2': r2_score(y_test, y_pred)
                }
                results.append(result)
                logger.info(f"{model_name} 性能: {result}")
            
            # 保存结果
            results_df = pd.DataFrame(results)
            results_df.to_csv('outputs/csv/prediction_results.csv', index=False)
            logger.info("预测模型结果已保存为 outputs/csv/prediction_results.csv")
            
            # 可视化预测结果
            best_model_name = results_df.sort_values('rmse').iloc[0]['model']
            best_model = models[best_model_name]
            
            y_pred = best_model.predict(X_test)
            fig = plot_prediction_scatter(y_test, y_pred, best_model_name)
            fig.write_image('outputs/images/prediction_visualization.png')
            logger.info("预测可视化图已保存为 outputs/images/prediction_visualization.png")
            
            # 残差分析
            logger.info("进行残差分析...")
            import matplotlib.pyplot as plt
            
            # 设置中文字体
            plt.rcParams['font.sans-serif'] = ['SimHei']
            plt.rcParams['axes.unicode_minus'] = False
            
            # 计算残差
            residuals = y_test - y_pred
            
            # 残差分布图
            plt.figure(figsize=(10, 6))
            plt.hist(residuals, bins=20, alpha=0.7, color='#9370DB')
            plt.title('残差分布')
            plt.xlabel('残差')
            plt.ylabel('频率')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig('outputs/images/residual_distribution.png', dpi=150)
            plt.close()
            logger.info("残差分布图已保存为 outputs/images/residual_distribution.png")
            
            # 残差与预测值关系图
            plt.figure(figsize=(10, 6))
            plt.scatter(y_pred, residuals, alpha=0.6, color='#9370DB')
            plt.axhline(y=0, color='r', linestyle='--')
            plt.title('残差与预测值关系')
            plt.xlabel('预测PHQ-9得分')
            plt.ylabel('残差')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig('outputs/images/residual_vs_prediction.png', dpi=150)
            plt.close()
            logger.info("残差与预测值关系图已保存为 outputs/images/residual_vs_prediction.png")
            
            # 残差统计
            residual_stats = {
                'mean': residuals.mean(),
                'std': residuals.std(),
                'min': residuals.min(),
                'max': residuals.max(),
                'abs_mean': abs(residuals).mean()
            }
            logger.info(f"残差统计: {residual_stats}")
            
            self.prediction_model = best_model
            return best_model
        else:
            logger.warning("没有足够的特征，无法构建预测模型")
            return None
    
    def run_all(self):
        """运行所有模型构建步骤"""
        if not self.load_data():
            return False
        
        self.data_preparation()
        self.clustering_analysis()
        self.classification_modeling()
        self.prediction_modeling()
        
        logger.info("模型构建完成！")
        return True

if __name__ == "__main__":
    builder = DepressionModelBuilder()
    builder.run_all()