import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import os
import logging
import json
import time
from model_building import DepressionModelBuilder


def _inject_js(code):
    """通过 iframe 注入 JS，避免 DOM removeChild 冲突"""
    components.html(f"<script>{code}</script>", height=0)
from visualization import (
    plot_phq9_distribution,
    plot_depression_severity_pie,
    plot_symptom_correlation,
    plot_symptom_combination,
    plot_time_series_trend,
    plot_demographic_analysis,
    plot_model_f1_bar,
    plot_cv_f1_bar,
    plot_prediction_cv_bar,
    generate_wordcloud
)

# 缓存数据加载函数
@st.cache_data(ttl=600)  # 缓存10分钟，云端更快刷新
def load_depression_data():
    """加载抑郁症状数据"""
    try:
        return pd.read_csv("outputs/csv/preprocessed_depression_data.csv", low_memory=False)
    except Exception as e:
        logging.error(f"加载抑郁症状数据失败: {e}")
        return None

@st.cache_data(ttl=3600)  # 缓存1小时
def load_suicide_data():
    """加载自杀检测数据"""
    try:
        return pd.read_csv("outputs/csv/preprocessed_suicide_data.csv", low_memory=False)
    except Exception as e:
        logging.error(f"加载自杀检测数据失败: {e}")
        return None

@st.cache_data(ttl=3600)  # 缓存1小时
def load_classification_results():
    """加载分类模型结果"""
    try:
        return pd.read_csv("outputs/csv/classification_results.csv")
    except Exception as e:
        logging.error(f"加载分类模型结果失败: {e}")
        return None

@st.cache_data(ttl=3600)  # 缓存1小时
def load_prediction_results():
    """加载预测模型结果"""
    try:
        return pd.read_csv("outputs/csv/prediction_results.csv")
    except Exception as e:
        logging.error(f"加载预测模型结果失败: {e}")
        return None

# ── 模块级 session_state 初始化（import 时立即执行）──
_SESSION_DEFAULTS = {
    'logged_in': False, 'username': None,
    'cluster_tsne_result': None, 'cluster_labels': None, 'cluster_built': False,
    'classification_cv_results': None, 'classification_results': None,
    'classification_importance': None, 'feature_importance': None, 'classification_built': False,
    'prediction_cv_results': None, 'prediction_results': None, 'prediction_built': False,
    'current_page': '🔬 聚类相关',
}
for _k, _v in _SESSION_DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v


def _ensure_dirs():
    os.makedirs('outputs/csv', exist_ok=True)
    os.makedirs('outputs/images', exist_ok=True)


# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def init_page_style():
    """初始化页面样式"""
    # 设置页面背景和样式
    st.markdown("""
    <style>
        body {
            background: linear-gradient(135deg, #F5F0FF, #EDE0F8);
            background-attachment: fixed;
            color: #333333;
        }
        .stApp {
            background: linear-gradient(135deg, #F5F0FF 0%, #E8DCF8 30%, #DCD0F0 60%, #EDE0F8 100%);
            background-attachment: fixed;
            padding-top: 0 !important;
        }
        .block-container { padding-top: 0.8rem !important; max-width: 90% !important; }
        h1, h2, h3, h4, h5, h6, p { color: #333333 !important; }
        h2 { margin-top: 0 !important; margin-bottom: 0.5rem !important; line-height: 1.2 !important; font-size: 1.5rem !important; }
        p { margin-top: 0.5rem !important; margin-bottom: 1rem !important; line-height: 1.4 !important; }

        .stButton > button {
            background-color: #9370DB !important;
            color: white !important;
            border-radius: 8px !important;
            padding: 8px 16px !important;
        }
        .stButton > button:hover { background-color: #7B68EE !important; }

        header { display: none !important; }

        .stRadio { margin-top: 10px !important; }
        .stRadio > div > div { margin-bottom: 10px !important; }
        .stRadio label { color: #444444 !important; }
        .stRadio [role="radio"]:checked { background-color: #9370DB !important; border-color: #9370DB !important; }
        .stRadio [role="radio"]:checked + label { color: #333333 !important; font-weight: 500 !important; }

        .dataframe th { background-color: #E6E6FA !important; color: #333 !important; text-align: center !important; }
        .dataframe td { border: 1px solid #E0E0E0 !important; text-align: center !important; }

        .stTabs [data-baseweb="tab-list"] { background-color: #9370DB !important; border: none !important; }
        .stTabs [data-baseweb="tab"] { color: white !important; font-weight: 500 !important; }
        .stTabs [data-baseweb="tab"]:hover { background-color: #7B68EE !important; }
        .stTabs [data-baseweb="tab"][aria-selected="true"] { background-color: #7B68EE !important; }

        hr { margin: 16px 0 !important; }
    </style>
    """, unsafe_allow_html=True)
    
    # 确保侧边栏固定显示且无法隐藏的样式
    st.markdown("""
    <style>
        /* 强制侧边栏始终展开 */
        [data-testid="stSidebar"] {
            min-width: 220px !important;
            max-width: 220px !important;
            background: linear-gradient(180deg, #F5F0FF 0%, #EDE0F8 50%, #E0D4F5 100%) !important;
        }

        /* 侧边栏内容区域 */
        [data-testid="stSidebarContent"] {
            background: transparent !important;
        }

        /* 导航栏目 */
        [data-testid="stSidebarNav"] {
            background: transparent !important;
        }

        [data-testid="stSidebarNav"] ul {
            background: transparent !important;
        }

        [data-testid="stSidebarNav"] a, [data-testid="stSidebarNav"] span {
            color: #333333 !important;
        }

        [data-testid="stSidebarNav"] li:hover {
            background-color: rgba(147, 112, 219, 0.1) !important;
        }

        [data-testid="stSidebarNav"] li[aria-current="page"] {
            background-color: #9370DB !important;
        }

        [data-testid="stSidebarNav"] li[aria-current="page"] span {
            color: white !important;
        }

        /* 侧边栏所有文字 */
        [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3, [data-testid="stSidebar"] p,
        [data-testid="stSidebar"] label, [data-testid="stSidebar"] div, [data-testid="stSidebar"] span {
            color: #333333 !important;
        }

        /* 侧边栏分割线 */
        [data-testid="stSidebar"] hr {
            border-color: rgba(147, 112, 219, 0.2) !important;
        }

        /* 退出登录按钮 */
        [data-testid="stSidebar"] .stButton > button {
            background-color: #9370DB !important;
            color: white !important;
        }

        /* 隐藏侧边栏折叠按钮 */
        [data-testid="stSidebarCollapseButton"] {
            display: none !important;
        }

        /* 侧边栏内容区域 */
        .css-1d391kg {
            display: block !important;
            visibility: visible !important;
            background: transparent !important;
        }
    </style>
    """, unsafe_allow_html=True)

def render_sidebar():
    """渲染侧边栏"""
    # 顶部版权信息
    st.markdown("### 🧠 抑郁症多维度心理特征分析系统")

    # 侧边栏用户信息显示
    st.sidebar.success(f"👤 {st.session_state.get('username', '用户')}")

    # 退出登录按钮
    if st.sidebar.button("退出登录", key="logout_button"):
        st.session_state.logged_in = False
        st.session_state.username = None
        st.rerun()

    # 侧边栏导航菜单
    st.sidebar.title("导航菜单")

    # 初始化页面状态
    if 'current_page' not in st.session_state:
        st.session_state.current_page = "🔬 聚类相关"

    nav_options = ["🔬 聚类相关", "🧪 分类模型相关", "📈 预测模型相关", "📊 应用内可视化"]
    default_idx = nav_options.index(st.session_state.get('current_page', nav_options[0])) if st.session_state.get('current_page', nav_options[0]) in nav_options else 0
    page = st.sidebar.radio("选择功能", nav_options, index=default_idx)
    st.session_state.current_page = page
    
    return page

# 检查登录状态的函数（用于功能访问控制）
def check_login():
    if not st.session_state.get('logged_in', False):
        st.warning("请先登录，请返回登录页面")
        st.stop()
    return True

def render_clustering_page():
    """渲染聚类相关页面"""
    if not check_login():
        st.stop()

    # 操作区
    with st.container():
        st.markdown("### 🔧 聚类模型构建")
        st.write("该功能将对抑郁症状数据进行聚类分析，发现抑郁症症状亚型，帮助理解不同类型的抑郁症状模式。")

        # 重新聚类按钮 -- 使用内存对象，不走 subprocess
        if st.button("重新聚类分析"):
            with st.spinner("正在构建聚类模型..."):
                start_time = time.time()
                try:
                    builder = DepressionModelBuilder()
                    if not builder.load_data():
                        st.error("加载数据失败")
                        st.stop()
                    builder.data_preparation()
                    tsne_result, kmeans_labels = builder.clustering_analysis()
                    st.session_state.cluster_tsne_result = tsne_result
                    st.session_state.cluster_labels = kmeans_labels
                    st.session_state.cluster_built = True
                    st.success("聚类分析完成！")
                    st.info(f"聚类分析耗时: {time.time() - start_time:.2f}秒")
                    st.rerun()
                except Exception as e:
                    st.error(f"聚类分析失败: {str(e)[:500]}")

    # 结果展示区
    st.markdown("---")
    st.markdown("### 📊 聚类分析结果")

    tsne_image_path = "outputs/images/clustering_visualization.png"
    evaluation_image_path = "outputs/images/clustering_evaluation.png"

    # t-SNE可视化 —— 始终显示预生成静态图，不随重新聚类变化
    if os.path.exists(tsne_image_path):
        st.subheader("t-SNE聚类可视化分析")
        with st.expander("图表解释"):
            st.write("t-SNE (t-distributed Stochastic Neighbor Embedding) 是一种非线性降维技术，专门用于将高维数据投影到二维空间，实现聚类/分类效果的可视化。")
            st.write("- **坐标轴**：x轴为t-SNE 1，y轴为t-SNE 2")
            st.write("- **颜色**：不同颜色代表不同的聚类标签")
            st.write("- **距离**：点之间的距离表示样本特征的相似性")
        st.image(tsne_image_path, use_container_width=True)
    else:
        st.info("尚未运行聚类分析，请点击上方「重新聚类分析」按钮生成结果。")

    # 显示聚类评估
    if os.path.exists(evaluation_image_path):
        st.subheader("聚类评估")
        with st.expander("图表解释"):
            st.write("聚类评估展示了不同聚类算法的性能指标。")
            st.write("- **指标**：包括轮廓系数、Calinski-Harabasz指数、Davies-Bouldin指数等")
            st.write("- **数值含义**：轮廓系数越接近1越好，Calinski-Harabasz指数越大越好，Davies-Bouldin指数越小越好")
        st.image(evaluation_image_path, use_container_width=True)
    else:
        st.warning("聚类评估图片不存在")
    
    # 簇画像分析
    try:
        # 尝试加载聚类结果文件
        clustering_file = "outputs/csv/clustering_results_sample.csv"
        cluster_profiles_file = "outputs/csv/cluster_profiles.csv"
        
        if os.path.exists(clustering_file):
            suicide_df = pd.read_csv(clustering_file, low_memory=False)
        elif os.path.exists(cluster_profiles_file):
            # 尝试加载簇画像文件
            suicide_df = pd.read_csv(cluster_profiles_file, low_memory=False)
        else:
            # 如果聚类结果文件不存在，尝试加载原始数据
            suicide_df = load_suicide_data()
        
        # ── 重新聚类结果（放在最下方，与上方原始图对比）──
        if st.session_state.get('cluster_tsne_result') is not None and st.session_state.get('cluster_labels') is not None:
            st.markdown("---")
            st.subheader("🔄 重新聚类结果")
            with st.expander("图表解释"):
                st.write("点击「重新聚类分析」后生成的新聚类结果。上方为原始预生成图，下方为实时计算结果，可对比查看。")
            from visualization import plot_tsne_clustering
            fig_new = plot_tsne_clustering(
                st.session_state.cluster_tsne_result,
                st.session_state.cluster_labels
            )
            st.plotly_chart(fig_new, use_container_width=True, config={'scrollZoom': True, 'displayModeBar': True})

        # 垂直堆叠布局
        # 簇画像分析
        st.subheader("簇画像分析")
        
        # 计算各簇的统计信息
        if suicide_df is not None and 'optimized_cluster' in suicide_df.columns:
            # 过滤掉NaN值
            valid_data = suicide_df.dropna(subset=['optimized_cluster'])
            
            if len(valid_data) > 0:
                # 计算各簇的统计信息
                cluster_profiles = valid_data.groupby('optimized_cluster').agg({
                    'text_length': 'mean',
                    'suicide_keyword_count': 'mean',
                    'class_label': 'mean'  # 自杀倾向比例
                }).reset_index()
                
                # 重命名列
                cluster_profiles.columns = ['簇', '平均文本长度', '平均自杀关键词数', '自杀倾向比例']
                
                # 确保所有簇（0、1、2）都存在
                all_clusters = [0, 1, 2]
                existing_clusters = cluster_profiles['簇'].tolist()
                
                # 添加缺失的簇
                for cluster in all_clusters:
                    if cluster not in existing_clusters:
                        new_row = {
                            '簇': cluster,
                            '平均文本长度': 0,
                            '平均自杀关键词数': 0,
                            '自杀倾向比例': 0
                        }
                        cluster_profiles = pd.concat([cluster_profiles, pd.DataFrame([new_row])], ignore_index=True)
                
                # 按簇排序
                cluster_profiles = cluster_profiles.sort_values('簇')
                
                st.dataframe(cluster_profiles, width="stretch")

                # ═══ 临床解读 ═══
                st.markdown("---")
                st.subheader("📋 各簇临床特征解读")
                # 根据数据动态生成解读
                interpretations = []
                for _, row in cluster_profiles.iterrows():
                    c = int(row['簇'])
                    length = row['平均文本长度']
                    keywords = row['平均自杀关键词数']
                    ratio = row['自杀倾向比例']
                    if ratio > 1.0:
                        ratio = ratio / (1 + ratio)  # 归一化防止异常值

                    if ratio > 0.5:
                        risk_level = "高风险"
                        desc = f"**簇{c}**：文本较短 (均长 {length:.0f} 字符) 但自杀关键词密度高 ({keywords:.1f} 个/条)，自杀倾向比例达 {ratio:.1%}。"
                        if length < 80:
                            desc += "短文本 + 高关键词密度提示**急性危机表达**——用户直接、简短地表达自杀意图，需立即干预。"
                        else:
                            desc += "该群体表现为**高自杀风险**，建议优先进行心理危机评估和干预。"
                    elif ratio > 0.3:
                        risk_level = "中风险"
                        desc = f"**簇{c}**：文本长度中等 (均长 {length:.0f} 字符)，自杀关键词 {keywords:.1f} 个/条，自杀倾向比例 {ratio:.1%}。"
                        desc += "该群体处于**中度风险区间**，可能伴随抑郁情绪表达但未达到急性危机程度，建议定期随访监测。"
                    else:
                        risk_level = "低风险"
                        desc = f"**簇{c}**：文本篇幅较长 (均长 {length:.0f} 字符)，自杀关键词少 ({keywords:.1f} 个/条)，自杀倾向比例仅 {ratio:.1%}。"
                        desc += "长文本 + 低风险组合提示该群体倾向于**叙述性表达**，更可能是在讨论或转述而非自述。属低风险群体，可进行常规心理健康教育。"

                    interpretations.append((risk_level, desc))

                # 按风险等级排列显示
                for risk_level, desc in sorted(interpretations):
                    if risk_level == "高风险":
                        st.error(desc)
                    elif risk_level == "中风险":
                        st.warning(desc)
                    else:
                        st.info(desc)

                st.markdown("---")
                st.markdown("\n")

                # 各簇自杀倾向比例
                st.subheader("各簇自杀倾向比例对比")
                import plotly.express as px
                # 确保簇是分类类型
                cluster_profiles['簇'] = cluster_profiles['簇'].astype(str)
                # 淡紫色渐变色系
                purple_colors = ['#9370DB', '#B19CD9', '#D8C4E8']
                fig = px.bar(cluster_profiles, x='簇', y='自杀倾向比例', 
                            title='各簇自杀倾向比例对比', 
                            color='簇', 
                            color_discrete_sequence=purple_colors,
                            template="plotly_white")
                fig.update_layout(
                    xaxis_title='簇',
                    yaxis_title='自杀倾向比例',
                    yaxis_range=[0, 1],
                    title=dict(
                        x=0.5,  # 居中
                        font=dict(
                            size=16,
                            color="#333333"
                        )
                    ),
                    plot_bgcolor='#F8F8FF',
                    paper_bgcolor='#F8F8FF',
                    showlegend=False  # 去掉图例
                )
                # 添加白色细边框
                fig.update_traces(
                    marker=dict(
                        line=dict(
                            width=1,
                            color="white"
                        )
                    )
                )
                # 自定义悬停提示
                fig.update_traces(
                    hovertemplate="簇编号: %{x}<br>自杀倾向比例: %{y:.4f}<extra></extra>"
                )
                st.plotly_chart(fig, width="stretch", height=500, config={'scrollZoom': True, 'displayModeBar': True})
                
                # 增加间距
                st.markdown("\n")
                
                # 添加词云分析
                st.subheader("各簇词云分析")
                # 检查词云图是否存在
                wordcloud_path = "outputs/images/suicide_wordcloud.png"
                if os.path.exists(wordcloud_path):
                    st.image(wordcloud_path, use_container_width=True)
                else:
                    st.info("词云分析功能正在开发中...")
                
                # 添加簇特征深度分析
                st.subheader("簇特征深度分析")
                # 计算各簇的详细统计信息
                detailed_stats = valid_data.groupby('optimized_cluster').agg({
                    'text_length': ['mean', 'std'],
                    'suicide_keyword_count': ['mean', 'std'],
                    'class_label': ['mean', 'count']
                }).round(4)
                
                # 重命名列
                detailed_stats.columns = ['平均文本长度', '文本长度标准差', '平均关键词数', '关键词数标准差', '自杀倾向比例', '样本数']
                st.dataframe(detailed_stats, width="stretch")
            else:
                st.info("没有有效的聚类结果，请点击「重新聚类分析」按钮生成。")
        else:
            st.info("暂无聚类结果数据，请点击「重新聚类分析」按钮生成。")
        

    except Exception as e:
        st.error(f"加载聚类结果失败: {e}")
        # 提供更详细的错误信息
        import traceback
        st.exception(traceback.format_exc())

def render_classification_page():
    """渲染分类模型相关页面"""
    if not check_login():
        st.stop()

    # 操作区
    with st.container():
        st.markdown("### 🧪 分类模型构建")
        if st.button("重新构建分类模型"):
            with st.spinner("正在构建分类模型..."):
                start_time = time.time()
                try:
                    builder = DepressionModelBuilder()
                    if not builder.load_data():
                        st.error("加载数据失败")
                        st.stop()
                    builder.data_preparation()
                    builder.classification_modeling()
                    cv_path = "outputs/csv/classification_cv_results.csv"
                    results_path = "outputs/csv/classification_results.csv"
                    importance_path = "outputs/csv/feature_importance.csv"
                    if os.path.exists(cv_path):
                        st.session_state.classification_cv_results = pd.read_csv(cv_path)
                    if os.path.exists(results_path):
                        st.session_state.classification_results = pd.read_csv(results_path)
                    if os.path.exists(importance_path):
                        st.session_state.classification_importance = pd.read_csv(importance_path)
                    st.session_state.classification_built = True
                    st.success("分类模型构建完成！")
                    st.info(f"分类模型构建耗时: {time.time() - start_time:.2f}秒")
                    st.rerun()
                except Exception as e:
                    st.error(f"分类模型构建失败: {e}")

    # 结果展示区
    st.markdown("---")
    st.markdown("### 📊 分类模型结果")

    cv_results = st.session_state.get('classification_cv_results')
    if cv_results is None:
        cv_path = "outputs/csv/classification_cv_results.csv"
        if os.path.exists(cv_path):
            cv_results = pd.read_csv(cv_path)

    class_results = st.session_state.get('classification_results')
    if class_results is None:
        results_path = "outputs/csv/classification_results.csv"
        if os.path.exists(results_path):
            class_results = pd.read_csv(results_path)

    importance_df = st.session_state.get('classification_importance')
    if importance_df is None:
        imp_path = "outputs/csv/feature_importance.csv"
        if os.path.exists(imp_path):
            importance_df = pd.read_csv(imp_path)

    confusion_matrix_path = "outputs/images/confusion_matrix.png"

    if class_results is None and cv_results is None and importance_df is None:
        # 没有任何结果数据
        if os.path.exists(confusion_matrix_path):
            st.subheader("混淆矩阵")
            st.image(confusion_matrix_path, use_container_width=True)
        st.info("尚未运行分类模型构建，请点击上方「重新构建分类模型」按钮生成结果。")
        return

    # ── 指标仪表盘 ──
    if cv_results is not None or class_results is not None:
        best_f1 = None
        best_auc = None
        best_model_name = ""
        if cv_results is not None and len(cv_results) > 0:
            best_cv = cv_results.loc[cv_results['f1_mean'].idxmax()]
            best_f1 = best_cv['f1_mean']
            best_auc = best_cv['auc_mean']
            best_model_name = best_cv['model']
        elif class_results is not None and len(class_results) > 0:
            non_bl = class_results[class_results['model'] != 'Baseline (Random)']
            if len(non_bl) > 0:
                best = non_bl.loc[non_bl['f1'].idxmax()]
                best_f1 = best['f1']
                best_auc = best.get('auc', 0)
                best_model_name = best['model']

        if best_f1 is not None:
            cols = st.columns(3)
            with cols[0]:
                st.metric("最佳 F1 分数", f"{best_f1:.4f}", help=best_model_name)
            with cols[1]:
                st.metric("最佳 AUC", f"{best_auc:.4f}", help="ROC 曲线下面积")
            with cols[2]:
                st.metric("对比模型数", "3", help="LR · RF · XGBoost")

    # 使用标签页布局
    tab1, tab2, tab3 = st.tabs(["交叉验证结果", "最终测试集结果", "特征重要性数据"])

    with tab1:
        st.subheader("5折交叉验证结果")
        with st.expander("表格解释"):
            st.write("此表格展示了5折交叉验证的结果，更可靠地评估模型性能。")
            st.write("- **f1_mean**：5折交叉验证F1分数的均值\n- **f1_std**：5折交叉验证F1分数的标准差（越小表示模型越稳定）\n- **auc_mean**：5折交叉验证AUC的均值\n- **auc_std**：5折交叉验证AUC的标准差")

        if cv_results is not None:
            st.dataframe(cv_results, use_container_width=True)
            st.subheader("交叉验证F1分数（带误差线）")
            with st.expander("图表解释"):
                st.write("此图展示了5折交叉验证的F1分数均值和标准差。")
                st.write("- **柱子高度**：F1分数均值（越高越好）\n- **误差线**：标准差（越短表示模型越稳定）")
            fig = plot_cv_f1_bar(cv_results)
            st.plotly_chart(fig, use_container_width=True, config={'scrollZoom': True, 'displayModeBar': True})
        else:
            st.info("暂无数据，请运行模型构建。")

    with tab2:
        st.subheader("最终测试集性能对比")
        with st.expander("表格解释"):
            st.write("此表格展示了在最终测试集上的模型性能指标。")
            st.write("- **准确率（accuracy）**：正确预测的样本数占总样本数的比例\n- **精确率（precision）**：预测为正例的样本中实际为正例的比例\n- **召回率（recall）**：实际为正例的样本中被正确预测的比例\n- **F1分数**：精确率和召回率的调和平均值\n- **AUC**：ROC曲线下面积")

        if class_results is not None:
            st.dataframe(class_results, use_container_width=True)
            st.subheader("各模型F1分数（最终测试集）")
            with st.expander("图表解释"):
                st.write("F1分数是模型性能的综合指标，平衡了精确率和召回率。")
                st.write("- **数值越高**：模型性能越好\n- **XGBoost**：结合TF-IDF文本特征后通常表现最佳")
            fig = plot_model_f1_bar(class_results)
            st.plotly_chart(fig, use_container_width=True, config={'scrollZoom': True, 'displayModeBar': True})
        else:
            st.info("暂无数据，请运行模型构建。")

        if os.path.exists(confusion_matrix_path):
            st.subheader("混淆矩阵")
            st.image(confusion_matrix_path, use_container_width=True)

    with tab3:
        st.subheader("特征重要性数据")

        if importance_df is not None:
            total_features = len(importance_df)

            # ── 视图切换 ──
            view_mode = st.radio(
                "选择显示视图",
                ["原始词条视图", "聚合类别视图"],
                horizontal=True,
                help="原始词条：逐词显示；聚合类别：按词类（死亡/否定/情感/第一人称/统计特征/其他）汇总"
            )

            if view_mode == "原始词条视图":
                with st.expander("表格解释"):
                    st.write("特征重要性数据展示了每个特征对模型预测的具体贡献值。")
                    st.write("- **特征名称**：输入特征的名称（如 tfidf_xxx 为TF-IDF词条）\n- **重要性分数**：特征对模型预测的贡献程度\n- **排序**：按重要性从高到低排序")

                max_display = st.slider(
                    "展示特征数（Top N）", 10, min(200, total_features), 20, 10,
                    help=f"当前共{total_features}个特征，显示Top N以避免卡顿"
                )
                top_n = importance_df.head(max_display).copy()
                top_n['feature'] = top_n['feature'].apply(
                    lambda x: x.replace('tfidf_', '') if isinstance(x, str) and x.startswith('tfidf_') else x
                )
                from visualization import plot_feature_importance
                fig = plot_feature_importance(top_n, "Best Model")
                st.plotly_chart(fig, use_container_width=True, config={'scrollZoom': True, 'displayModeBar': True})
                st.subheader(f"特征重要性原始数据（展示前{max_display}/{total_features}条）")
                st.dataframe(top_n, use_container_width=True)

            else:  # 聚合类别视图
                with st.expander("图表解释"):
                    st.write("将 TF-IDF 词条特征按语义类别聚合后的重要性总和。")
                    st.write("- **死亡相关词**：die, kill, end, death, suicide, dead 等")
                    st.write("- **否定词**：dont, cant, not, never, nothing 等")
                    st.write("- **第一人称代词**：i, my, me, myself 等")
                    st.write("- **情感/痛苦词**：feel, pain, sad, hate, hurt, alone 等")
                    st.write("- **统计特征**：文本长度、自杀关键词数量、第一人称占比、情感得分、消极词占比、情绪强度")
                    st.write("- **其他词**：未归入以上类别的所有 TF-IDF 词条")

                from visualization import aggregate_tfidf_importance, plot_category_importance
                agg_df = aggregate_tfidf_importance(importance_df)
                fig_agg = plot_category_importance(agg_df)
                st.plotly_chart(fig_agg, use_container_width=True, config={'scrollZoom': True, 'displayModeBar': True})
                st.subheader("聚合类别数据表")
                st.dataframe(agg_df, use_container_width=True)
        else:
            with st.expander("表格解释"):
                st.write("特征重要性数据展示了每个特征对模型预测的具体贡献值。")
            st.info("暂无数据，请运行模型构建。")

def render_prediction_page():
    """渲染预测模型相关页面"""
    if not check_login():
        st.stop()

    # 操作区
    with st.container():
        st.markdown("### 📈 预测模型构建")
        st.write("该功能将构建预测模型，预测抑郁症状态变化趋势，帮助提前识别风险。")
        if st.button("重新构建预测模型"):
            with st.spinner("正在构建预测模型..."):
                start_time = time.time()
                try:
                    builder = DepressionModelBuilder()
                    if not builder.load_data():
                        st.error("加载数据失败")
                        st.stop()
                    builder.data_preparation()
                    builder.prediction_modeling()
                    cv_path = "outputs/csv/prediction_cv_results.csv"
                    results_path = "outputs/csv/prediction_results.csv"
                    if os.path.exists(cv_path):
                        st.session_state.prediction_cv_results = pd.read_csv(cv_path)
                    if os.path.exists(results_path):
                        st.session_state.prediction_results = pd.read_csv(results_path)
                    st.session_state.prediction_built = True
                    st.success("预测模型构建完成！")
                    st.info(f"预测模型构建耗时: {time.time() - start_time:.2f}秒")
                    st.rerun()
                except Exception as e:
                    st.error(f"预测模型构建失败: {e}")

    # 结果展示区
    st.markdown("---")
    st.markdown("### 📊 预测模型结果")

    # 优先从 session_state 读取
    cv_results = st.session_state.get('prediction_cv_results')
    if cv_results is None:
        cv_path = "outputs/csv/prediction_cv_results.csv"
        if os.path.exists(cv_path):
            cv_results = pd.read_csv(cv_path)

    pred_results = st.session_state.get('prediction_results')
    if pred_results is None:
        results_path = "outputs/csv/prediction_results.csv"
        if os.path.exists(results_path):
            pred_results = pd.read_csv(results_path)

    # 静态图片路径
    prediction_viz_path = "outputs/images/prediction_visualization.png"
    residual_dist_path = "outputs/images/residual_distribution.png"
    residual_vs_pred_path = "outputs/images/residual_vs_prediction.png"

    if cv_results is None and pred_results is None:
        st.info("尚未运行预测模型构建，请点击上方「重新构建预测模型」按钮生成结果。")
        return

    pred_tab1, pred_tab2 = st.tabs(["交叉验证结果", "最终测试集结果"])

    with pred_tab1:
        st.subheader("5折交叉验证结果")
        with st.expander("表格解释"):
            st.write("此表格展示了5折交叉验证的结果，更可靠地评估模型性能。")
            st.write("- **mae_mean/std**：平均绝对误差的均值和标准差（越小越好）\n- **rmse_mean/std**：均方根误差的均值和标准差（越小越好）\n- **r2_mean/std**：决定系数的均值和标准差（越接近1越好）")

        if cv_results is not None:
            cv_column_mapping = {
                'model': '模型名称', 'mae_mean': 'MAE均值', 'mae_std': 'MAE标准差',
                'rmse_mean': 'RMSE均值', 'rmse_std': 'RMSE标准差',
                'r2_mean': 'R²均值', 'r2_std': 'R²标准差'
            }
            cv_display = cv_results.rename(columns=cv_column_mapping)
            for col in ['MAE均值', 'MAE标准差', 'RMSE均值', 'RMSE标准差', 'R²均值', 'R²标准差']:
                if col in cv_display.columns:
                    cv_display[col] = cv_display[col].round(4)
            st.dataframe(cv_display, use_container_width=True)

            st.subheader("交叉验证性能对比（带误差线）")
            fig = plot_prediction_cv_bar(cv_results)
            st.plotly_chart(fig, use_container_width=True, config={'scrollZoom': True, 'displayModeBar': True})
        else:
            st.info("暂无数据，请运行模型构建。")

    with pred_tab2:
        st.subheader("最终测试集模型性能")
        with st.expander("表格解释"):
            st.write("此表格展示了在最终测试集上的模型性能指标。")
            st.write("- **MAE**：预测值与实际值绝对误差的平均值\n- **RMSE**：预测值与实际值误差的平方根\n- **R²**：模型解释数据变异的比例，越接近1越好")

        if pred_results is not None:
            column_mapping = {
                'model': '模型名称', 'mae': '平均绝对误差(MAE)',
                'rmse': '均方根误差(RMSE)', 'r2': '决定系数(R²)'
            }
            pred_display = pred_results.rename(columns=column_mapping)
            for col in ['平均绝对误差(MAE)', '均方根误差(RMSE)']:
                if col in pred_display.columns:
                    pred_display[col] = pred_display[col].round(6)
            if '决定系数(R²)' in pred_display.columns:
                pred_display['决定系数(R²)'] = pred_display['决定系数(R²)'].round(4)

            st.dataframe(pred_display, use_container_width=True)

            # ── 玻璃仪表盘：最佳模型 R² ──
            non_baseline = pred_results[pred_results['model'] != 'Baseline (Mean)']
            if len(non_baseline) > 0:
                best_row = non_baseline.loc[non_baseline['r2'].idxmax()]
                best_r2 = best_row['r2']
                best_name = best_row['model']
                r2_pct = best_r2 * 100
                best_mae = best_row['mae']
                best_rmse = best_row['rmse']

                r2_color = "#27AE60" if best_r2 >= 0.7 else ("#F39C12" if best_r2 >= 0.4 else "#E74C3C")
                cols = st.columns(3)
                with cols[0]:
                    st.metric("R² 决定系数", f"{best_r2:.4f}",
                             help=f"{best_name} · 解释 {r2_pct:.1f}% 变异")
                with cols[1]:
                    st.metric("MAE 平均绝对误差", f"{best_mae:.4f}",
                             help="预测值与真实值平均偏差")
                with cols[2]:
                    st.metric("RMSE 均方根误差", f"{best_rmse:.4f}",
                             help="对大误差更敏感")

                if best_r2 >= 0.7:
                    st.success(f"✅ 模型拟合效果**良好**：{best_name} 的 R²={best_r2:.4f}，具备临床参考价值。")
                elif best_r2 >= 0.4:
                    st.warning(f"⚠️ 模型拟合**中等**：R²={best_r2:.4f}。PHQ-9 受多因素影响，此结果符合预期。")
                else:
                    st.error(f"❌ 模型拟合**较弱**：R²={best_r2:.4f}，可能缺少非线性交互特征。")
        else:
            st.info("暂无数据，请运行模型构建。")

        # 显示静态图片
        if os.path.exists(prediction_viz_path):
            st.subheader("预测可视化")
            with st.expander("图表解释"):
                st.write("此图展示了模型预测值与实际值的对比。")
            st.image(prediction_viz_path, use_container_width=True)

        if os.path.exists(residual_dist_path):
            st.subheader("残差分布")
            st.image(residual_dist_path, use_container_width=True)

        if os.path.exists(residual_vs_pred_path):
            st.subheader("残差与预测值关系")
            st.image(residual_vs_pred_path, use_container_width=True)

def render_visualization_page():
    """渲染应用内可视化页面"""
    # 检查登录状态
    if not check_login():
        st.stop()
    
    st.subheader("数据可视化")
    try:
        # 加载预处理后的数据
        depression_df = load_depression_data()
        
        # 使用标签页布局
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["基本统计", "相关性分析", "文本分析", "趋势分析", "人口统计"])
        
        with tab1:
            # PHQ-9总分分布
            st.subheader("PHQ-9总分分布")
            # 添加解释下拉菜单
            with st.expander("图表解释"):
                st.write("PHQ-9是一个9项抑郁症筛查量表，总分范围为0-27分。")
                st.write("- **分数区间**：\n  - 0-4分：正常\n  - 5-9分：轻度抑郁\n  - 10-14分：中度抑郁\n  - 15-27分：重度抑郁\n- **分布形状**：理想情况下应该呈现正态分布或轻微右偏\n- **峰值位置**：反映样本的整体抑郁水平")
            fig = plot_phq9_distribution(depression_df)
            st.plotly_chart(fig, width="stretch", config={'scrollZoom': True, 'displayModeBar': True})
            
            # 抑郁严重程度饼图
            st.subheader("抑郁严重程度分布")
            # 添加解释下拉菜单
            with st.expander("图表解释"):
                st.write("此图展示了不同抑郁严重程度的样本比例。")
                st.write("- **颜色**：不同颜色代表不同的抑郁严重程度\n- **扇区大小**：反映各严重程度的样本占比\n- **诊断意义**：帮助了解样本的整体心理健康状况\n- **干预重点**：识别需要重点关注的群体")
            fig = plot_depression_severity_pie(depression_df)
            st.plotly_chart(fig, width="stretch", config={'scrollZoom': True, 'displayModeBar': True})
        
        with tab2:
            # 症状相关性热力图
            st.subheader("症状相关性热力图")
            # 添加解释下拉菜单
            with st.expander("图表解释"):
                st.write("此图展示了PHQ-9各症状之间的相关性。")
                st.write("- **颜色深浅**：颜色越深表示相关性越强\n- **对角线**：每个症状与自身的相关性为1\n- **症状关联**：显示哪些症状经常同时出现\n- **临床意义**：帮助理解抑郁症的症状集群\n- **PHQ项目**：\n  - phq1：兴趣减退\n  - phq2：情绪低落\n  - phq3：睡眠问题\n  - phq4：疲劳\n  - phq5：食欲变化\n  - phq6：自我否定\n  - phq7：注意力困难\n  - phq8：动作迟滞\n  - phq9：自杀倾向")
            phq_columns = [f"phq{i}" for i in range(1, 10)]
            corr_matrix = depression_df[phq_columns].corr()
            fig = plot_symptom_correlation(corr_matrix)
            st.plotly_chart(fig, width="stretch", config={'scrollZoom': True, 'displayModeBar': True})
            
            # 添加垂直间距
            st.markdown("\n\n")
            
            # 症状组合分析
            st.subheader("症状组合分析")
            # 添加解释下拉菜单
            with st.expander("图表解释"):
                st.write("此图展示了最常见的抑郁症状组合。")
                st.write("- **症状组合**：同时出现的症状组合\n- **出现频率**：展示各组合的出现次数\n- **临床意义**：识别典型的症状模式\n- **干预指导**：针对常见症状组合制定干预策略")
            try:
                fig = plot_symptom_combination(depression_df)
                st.plotly_chart(fig, width="stretch", config={'scrollZoom': True, 'displayModeBar': True})
            except Exception as e:
                st.error(f"生成症状组合分析图失败: {e}")
        
        with tab3:
            # 词云图 - 自杀倾向文本分析
            st.subheader("自杀倾向文本词云")
            # 添加解释下拉菜单
            with st.expander("图表解释"):
                st.write("词云展示了自杀倾向文本中最常见的词汇。")
                st.write("- **词大小**：词出现的频率越高，显示越大\n- **词颜色**：不同颜色区分不同类别的词汇\n- **文本来源**：来自有自杀倾向的用户发布的文本\n- **临床意义**：帮助识别自杀风险的语言特征\n- **干预提示**：识别高风险词汇，用于早期干预")
            try:
                wordcloud_path = "outputs/images/suicide_wordcloud.png"
                if not os.path.exists(wordcloud_path):
                    suicide_df = load_suicide_data()
                    if suicide_df is not None and 'class_label' in suicide_df.columns and 'text' in suicide_df.columns:
                        suicide_texts = suicide_df[suicide_df['class_label'] == 1]['text'].dropna().tolist()
                        if suicide_texts:
                            combined_text = ' '.join(suicide_texts[:500])
                            wordcloud_path = generate_wordcloud(combined_text)

                if os.path.exists(wordcloud_path):
                    st.image(wordcloud_path, use_container_width=True)
                else:
                    st.info("词云图暂未生成，请确保数据已预处理。")
            except Exception as e:
                st.error(f"生成词云失败: {e}")
        
        with tab4:
            # 时间序列趋势
            st.subheader("抑郁症状时间趋势")
            # 添加解释下拉菜单
            with st.expander("图表解释"):
                st.write("此图展示了抑郁症状随时间的变化趋势。")
                st.write("- **时间轴**：横轴表示时间（天数）\n- **纵轴**：纵轴表示PHQ-9总分或症状得分\n- **趋势线**：显示症状随时间的变化趋势\n- **临床意义**：帮助了解抑郁症的病程变化\n- **干预效果**：评估治疗或干预的效果")
            try:
                fig = plot_time_series_trend(depression_df)
                st.plotly_chart(fig, width="stretch", config={'scrollZoom': True, 'displayModeBar': True})
            except Exception as e:
                st.error(f"生成时间序列趋势图失败: {e}")
        
        with tab5:
            # 人口统计学分析
            st.subheader("人口统计学分析")
            # 添加解释下拉菜单
            with st.expander("图表解释"):
                st.write("此图展示了不同人口统计学特征的抑郁情况。")
                st.write("- **人口特征**：包括年龄、性别等\n- **抑郁分布**：不同人口特征的抑郁严重程度分布\n- **差异分析**：识别高风险人群\n- **干预指导**：针对不同人群制定个性化干预策略")
            try:
                fig = plot_demographic_analysis(depression_df)
                st.plotly_chart(fig, width="stretch", config={'scrollZoom': True, 'displayModeBar': True})
            except Exception as e:
                st.error(f"生成人口统计学分析图失败: {e}")
            
            # 情感得分与自杀倾向关系
            st.subheader("情感得分与自杀倾向关系")
            # 添加解释下拉菜单
            with st.expander("图表解释"):
                st.write("此图展示了文本情感得分与自杀倾向之间的关系。")
                st.write("- **散点**：每个点代表一个文本样本\n- **X轴**：情感得分（负值表示消极，正值表示积极）\n- **Y轴**：自杀倾向（0=非自杀，1=自杀）\n- **趋势线**：显示情感得分与自杀倾向的线性关系\n- **临床意义**：情感越消极，自杀倾向越高\n- **干预提示**：识别情感消极的高风险个体")
            try:
                # 加载自杀检测数据
                suicide_df = load_suicide_data()
                
                # 确保必要的列存在
                if suicide_df is not None and 'sentiment_score' in suicide_df.columns and 'class_label' in suicide_df.columns:
                    # 创建散点图
                    import plotly.express as px
                    import numpy as np
                    from sklearn.linear_model import LinearRegression
                    
                    # 准备数据
                    X = suicide_df['sentiment_score'].values.reshape(-1, 1)
                    y = suicide_df['class_label'].values
                    
                    # 计算线性回归
                    model = LinearRegression()
                    model.fit(X, y)
                    y_pred = model.predict(X)
                    
                    # 创建散点图
                    fig = px.scatter(
                        suicide_df,
                        x="sentiment_score",
                        y="class_label",
                        title="情感得分与自杀倾向关系",
                        labels={"sentiment_score": "情感得分", "class_label": "自杀倾向（0=非自杀, 1=自杀）"},
                        color="class_label",
                        color_continuous_scale=["#E6E6FA", "#9370DB"],
                        template="plotly_white"
                    )
                    
                    # 添加趋势线
                    fig.add_trace(px.line(
                        x=suicide_df['sentiment_score'],
                        y=y_pred,
                        color_discrete_sequence=["#9370DB"]
                    ).data[0])
                    
                    # 更新趋势线样式
                    fig.update_traces(
                        line=dict(width=2, dash="dash"),
                        selector=dict(mode="lines")
                    )
                    
                    # 优化布局
                    fig.update_layout(
                        hovermode="closest",
                        hoverlabel=dict(bgcolor="white", font_size=12, font_family="Arial"),
                        margin=dict(l=40, r=40, t=60, b=40),
                        plot_bgcolor='#F8F8FF',
                        paper_bgcolor='#F8F8FF'
                    )
                    
                    st.plotly_chart(fig, width="stretch", config={'scrollZoom': True, 'displayModeBar': True})
                else:
                    st.warning("数据中缺少情感得分或自杀倾向列")
            except Exception as e:
                st.error(f"生成情感得分与自杀倾向关系图失败: {e}")

    except Exception as e:
        st.error(f"加载数据失败: {e}")

# 主函数 - 用于模块导入时调用
def main():
    """主函数 - 当app.py作为模块被导入时调用"""
    _ensure_dirs()

    if not st.session_state.get('logged_in', False):
        st.warning("请先登录，请返回登录页面")
        st.stop()

    init_page_style()

    page = render_sidebar()

    if "聚类" in page:
        render_clustering_page()
    elif "分类" in page:
        render_classification_page()
    elif "预测" in page:
        render_prediction_page()
    elif "可视化" in page:
        render_visualization_page()

# 当直接运行app.py时，调用main函数
if __name__ == "__main__":
    main()
