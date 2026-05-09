import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import random
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# 设置全局随机种子，确保结果可复现
random.seed(42)
# 使用新的NumPy随机数生成器写法
rng = np.random.default_rng(seed=42)

# 设置默认模板
px.defaults.template = "plotly_white"

# 定义淡紫色系颜色
PURPLE_COLORS = {
    "light": "#E6E6FA",  # 淡紫色
    "medium": "#CBC3E3",  # 薰衣草紫
    "dark": "#9370DB",    # 藕荷紫
    "darker": "#7B68EE",  # 深藕荷紫
    "darkest": "#6A5ACD"   # 暗紫色
}

# 柱状图 - 各簇PHQ-9平均分
def plot_cluster_phq9_bar(cluster_profiles):
    """
    绘制各簇PHQ-9平均分柱状图
    
    参数:
        cluster_profiles: DataFrame - 包含簇信息的DataFrame
    
    返回:
        fig: Plotly Figure - 生成的图表
    """
    fig = px.bar(
        cluster_profiles,
        x="cluster",
        y="phq9_mean",
        title="各簇PHQ-9平均分",
        labels={"cluster": "簇", "phq9_mean": "PHQ-9平均分"},
        color="phq9_mean",
        color_continuous_scale=[PURPLE_COLORS["light"], PURPLE_COLORS["dark"]],
        template="plotly_white"
    )
    
    # 添加圆角
    fig.update_traces(marker=dict(line=dict(color=PURPLE_COLORS["darker"], width=1), opacity=0.8), selector=dict(type="bar"))
    
    # 添加交互功能
    fig.update_layout(
        hovermode="x unified",
        hoverlabel=dict(bgcolor="white", font_size=12, font_family="Arial"),
        margin=dict(l=40, r=40, t=60, b=40)
    )
    
    return fig

# 散点图 - t-SNE聚类可视化
def plot_tsne_clustering(tsne_result, labels):
    """
    绘制t-SNE聚类可视化散点图
    
    参数:
        tsne_result: numpy array - t-SNE降维结果
        labels: numpy array - 聚类标签
    
    返回:
        fig: Plotly Figure - 生成的图表
    """
    # 创建DataFrame
    df = pd.DataFrame({
        "tsne_1": tsne_result[:, 0],
        "tsne_2": tsne_result[:, 1],
        "cluster": labels[:len(tsne_result)],
        "index": range(len(tsne_result))  # 添加样本索引
    })
    
    # [改动点] 确保cluster列是字符串类型，避免Plotly将其当作连续数值从而渲染右侧长数字刻度
    df["cluster"] = df["cluster"].astype(str)

    # 簇标签映射
    cluster_labels = sorted(df["cluster"].unique(), key=lambda x: int(x) if x.isdigit() else x)
    cluster_mapping = {str(i): f"簇{i}" for i in range(len(cluster_labels))}
    df["cluster_name"] = df["cluster"].map(cluster_mapping)

    # 定义离散分类颜色（红、蓝、绿，确保与簇一一对应）
    discrete_colors = ["#E74C3C", "#3498DB", "#2ECC71"]
    color_discrete_map = {
        lbl: discrete_colors[i % len(discrete_colors)]
        for i, lbl in enumerate(cluster_labels)
    }

    fig = px.scatter(
        df,
        x="tsne_1",
        y="tsne_2",
        color="cluster_name",
        title="t-SNE 聚类可视化",
        labels={"tsne_1": "t-SNE 1", "tsne_2": "t-SNE 2", "cluster_name": "簇"},
        color_discrete_map=color_discrete_map,
        template="plotly_white"
    )
    
    # 添加交互功能
    fig.update_traces(
        marker=dict(
            size=6,  # 调整点大小
            opacity=0.7,  # 调整透明度
            line=dict(color="white", width=0.5)  # 细白边框
        ),
        selector=dict(type="scatter")
    )
    
    # 优化布局
    fig.update_layout(
        hovermode="closest",
        hoverlabel=dict(bgcolor="white", font_size=12, font_family="Arial"),
        margin=dict(l=60, r=40, t=80, b=60),
        title=dict(
            text="t-SNE 聚类可视化",
            x=0.5,
            y=0.95,
            font=dict(size=18, color="#333333")
        ),
        xaxis=dict(
            title=dict(text="t-SNE 1", font=dict(size=14, color="#333333")),
            tickfont=dict(size=12),
            showgrid=False
        ),
        yaxis=dict(
            title=dict(text="t-SNE 2", font=dict(size=14, color="#333333")),
            tickfont=dict(size=12),
            showgrid=False
        ),
        legend=dict(
            title=dict(text="簇", font=dict(size=14, color="#333333")),
            font=dict(size=12),
            bgcolor="white",
            bordercolor="#E0E0E0",
            borderwidth=1
        ),
        modebar_remove=["toImage", "sendDataToCloud"],
        plot_bgcolor="white",
        paper_bgcolor="white"
    )
    
    # 自定义悬停提示
    fig.update_traces(
        hovertemplate="簇: %{customdata[0]}<br>t-SNE 1: %{x:.2f}<br>t-SNE 2: %{y:.2f}<extra></extra>",
        customdata=df[["cluster_name"]].values
    )
    
    return fig

# 柱状图 - 各模型F1分数（最终测试集结果）
def plot_model_f1_bar(classification_results):
    """
    绘制各模型F1分数柱状图
    
    参数:
        classification_results: DataFrame - 包含模型性能的DataFrame
    
    返回:
        fig: Plotly Figure - 生成的图表
    """
    fig = px.bar(
        classification_results,
        x="model",
        y="f1",
        title="各模型F1分数（最终测试集）",
        labels={"model": "模型", "f1": "F1分数"},
        color="f1",
        color_continuous_scale=[PURPLE_COLORS["light"], PURPLE_COLORS["dark"]],
        template="plotly_white"
    )
    
    # 添加圆角
    fig.update_traces(marker=dict(line=dict(color=PURPLE_COLORS["darker"], width=1), opacity=0.8), selector=dict(type="bar"))
    
    # 添加交互功能
    fig.update_layout(
        hovermode="x unified",
        hoverlabel=dict(bgcolor="white", font_size=12, font_family="Arial"),
        margin=dict(l=40, r=40, t=60, b=40),
        xaxis=dict(tickangle=45)
    )
    
    return fig


# 柱状图 - 交叉验证F1分数（带误差线）
def plot_cv_f1_bar(cv_results_df):
    """
    绘制交叉验证F1分数柱状图（带误差线）
    
    参数:
        cv_results_df: DataFrame - 包含交叉验证结果的DataFrame
    
    返回:
        fig: Plotly Figure - 生成的图表
    """
    fig = go.Figure()
    
    # 添加柱状图
    fig.add_trace(go.Bar(
        x=cv_results_df['model'],
        y=cv_results_df['f1_mean'],
        error_y=dict(
            type='data',
            array=cv_results_df['f1_std'],
            visible=True,
            color=PURPLE_COLORS['darker'],
            thickness=1.5,
            width=8
        ),
        marker=dict(
            color=PURPLE_COLORS['dark'],
            line=dict(color=PURPLE_COLORS['darker'], width=1),
            opacity=0.8
        ),
        name='F1分数'
    ))
    
    # 更新布局
    fig.update_layout(
        title=dict(
            text="5折交叉验证F1分数（均值±标准差）",
            x=0.5,
            font=dict(size=18, color="#333333")
        ),
        xaxis=dict(
            title=dict(text="模型", font=dict(size=14)),
            tickangle=45,
            tickfont=dict(size=12)
        ),
        yaxis=dict(
            title=dict(text="F1分数", font=dict(size=14)),
            range=[0, 1],
            tickfont=dict(size=12)
        ),
        hovermode="x unified",
        hoverlabel=dict(bgcolor="white", font_size=12, font_family="Arial"),
        margin=dict(l=60, r=40, t=80, b=80),
        template="plotly_white",
        showlegend=False
    )
    
    return fig


# 分组柱状图 - 交叉验证各折结果对比
def plot_cv_folds_comparison(cv_results_df):
    """
    绘制交叉验证各折F1分数对比图
    
    参数:
        cv_results_df: DataFrame - 包含交叉验证详细结果的DataFrame
    
    返回:
        fig: Plotly Figure - 生成的图表
    """
    # 检查是否有各折的详细数据
    if 'f1_scores' not in cv_results_df.columns:
        # 如果没有详细数据，创建一个简单的提示图
        fig = go.Figure()
        fig.add_annotation(
            x=0.5, y=0.5,
            xref="paper", yref="paper",
            text="暂无各折详细数据",
            showarrow=False,
            font=dict(size=16, color="#666666")
        )
        fig.update_layout(
            title="交叉验证各折结果对比",
            template="plotly_white"
        )
        return fig
    
    fig = go.Figure()
    
    # 为每个模型添加折线
    colors = ['#9370DB', '#7B68EE', '#6A5ACD', '#5B4BC4']
    for idx, row in cv_results_df.iterrows():
        f1_scores = eval(row['f1_scores']) if isinstance(row['f1_scores'], str) else row['f1_scores']
        fig.add_trace(go.Scatter(
            x=list(range(1, len(f1_scores) + 1)),
            y=f1_scores,
            mode='lines+markers',
            name=row['model'],
            line=dict(color=colors[idx % len(colors)], width=2),
            marker=dict(size=8, color=colors[idx % len(colors)])
        ))
    
    fig.update_layout(
        title=dict(
            text="交叉验证各折F1分数对比",
            x=0.5,
            font=dict(size=18, color="#333333")
        ),
        xaxis=dict(
            title=dict(text="折数", font=dict(size=14)),
            tickmode='linear',
            tick0=1,
            dtick=1,
            tickfont=dict(size=12)
        ),
        yaxis=dict(
            title=dict(text="F1分数", font=dict(size=14)),
            range=[0, 1],
            tickfont=dict(size=12)
        ),
        hovermode="x unified",
        hoverlabel=dict(bgcolor="white", font_size=12, font_family="Arial"),
        margin=dict(l=60, r=40, t=80, b=60),
        template="plotly_white",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5
        )
    )
    
    return fig

# 柱状图 - 特征重要性
def plot_feature_importance(importance_df, model_name):
    """
    绘制特征重要性柱状图
    
    参数:
        importance_df: DataFrame - 包含特征重要性的DataFrame
        model_name: str - 模型名称
    
    返回:
        fig: Plotly Figure - 生成的图表
    """
    # 特征名映射（英文到中文）
    feature_mapping = {
        'text_length_scaled': '文本长度',
        'first_person_ratio_scaled': '第一人称占比',
        'sentiment_score_scaled': '情感得分',
        'negative_word_ratio_scaled': '消极词占比',
        'emotion_intensity_scaled': '情绪强度',
        'suicide_keyword_count_scaled': '自杀关键词数量'
    }
    
    # 替换特征名为中文，但保持已经是中文的特征名不变
    importance_df['feature'] = importance_df['feature'].apply(lambda x: feature_mapping.get(x, x))
    
    # 按重要性从高到低排序
    importance_df = importance_df.sort_values('importance', ascending=False)
    
    fig = px.bar(
        importance_df,
        x="importance",
        y="feature",
        orientation="h",
        title="SHAP特征重要性分析",
        labels={"importance": "平均SHAP值（对模型输出的影响）", "feature": "特征"},
        color_discrete_sequence=["#9370DB"],  # 使用固定的淡紫色
        template="plotly_white"
    )
    
    # 添加圆角和样式
    fig.update_traces(
        marker=dict(
            line=dict(color="#7B68EE", width=1), 
            opacity=0.8,
            # 添加圆角
            line_width=1,
            # 圆角设置
            cornerradius=4
        ), 
        selector=dict(type="bar")
    )
    
    # 添加交互功能
    fig.update_layout(
        hovermode="y",
        hoverlabel=dict(bgcolor="white", font_size=12, font_family="Arial"),
        margin=dict(l=100, r=40, t=80, b=40),
        # 标题居中
        title=dict(
            text="SHAP特征重要性分析",
            x=0.5,
            y=0.95,
            font=dict(
                size=18,
                color="#333333"
            )
        )
    )
    
    # 自定义悬停提示
    fig.update_traces(
        hovertemplate="特征: %{y}<br>平均SHAP值: %{x:.4f}<extra></extra>"
    )
    
    return fig

# 散点图 - 预测值 vs 实际值
def plot_prediction_scatter(y_test, y_pred, model_name):
    """
    绘制预测值 vs 实际值散点图
    
    参数:
        y_test: numpy array - 实际值
        y_pred: numpy array - 预测值
        model_name: str - 模型名称
    
    返回:
        fig: Plotly Figure - 生成的图表
    """
    from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
    
    df = pd.DataFrame({
        "实际值": y_test,
        "预测值": y_pred
    })
    
    # 计算评估指标
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    # 兼容旧版本scikit-learn，使用手动开平方
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    
    # 计算对角线
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    
    fig = go.Figure()
    
    # 添加散点
    fig.add_trace(go.Scatter(
        x=df["实际值"],
        y=df["预测值"],
        mode="markers",
        marker=dict(
            color=PURPLE_COLORS["dark"],  # 使用更深的紫色
            opacity=0.7,  # 调整透明度
            size=10,  # 增大点的大小
            line=dict(
                color="white",  # 白色边框
                width=1  # 边框宽度
            ),
            # 添加渐变色效果
            colorscale=[PURPLE_COLORS["light"], PURPLE_COLORS["dark"]]
        ),
        name="预测值",
        hovertext=[f"实际值: {x:.2f}<br>预测值: {y:.2f}" for x, y in zip(df["实际值"], df["预测值"])],
        # 添加动画效果
        hoverinfo="text"
    ))
    
    # 添加对角线
    fig.add_trace(go.Scatter(
        x=[min_val, max_val],
        y=[min_val, max_val],
        mode="lines",
        line=dict(
            color=PURPLE_COLORS["darkest"],  # 深紫色
            width=3,  # 增粗线条
            dash="dash"  # 使用标准虚线样式
        ),
        name="理想拟合线"
    ))
    
    # 更新布局
    fig.update_layout(
        # 美化标题
        title=dict(
            text=f"{model_name}：预测值 vs 实际值",
            x=0.5,  # 居中
            y=0.95,
            font=dict(
                size=20,  # 增大字号
                color="#333333",  # 深灰色
                family="Arial"
            )
        ),
        # 优化坐标轴
        xaxis=dict(
            title=dict(
                text="实际PHQ-9得分",
                font=dict(
                    size=14,
                    color="#666666"
                )
            ),
            range=[min_val - 1, max_val + 1],
            gridcolor="#F0F0F0",  # 浅灰色网格
            showline=True,
            linecolor="#E0E0E0",
            mirror=True
        ),
        yaxis=dict(
            title=dict(
                text="预测PHQ-9得分",
                font=dict(
                    size=14,
                    color="#666666"
                )
            ),
            range=[min_val - 1, max_val + 1],
            gridcolor="#F0F0F0",  # 浅灰色网格
            showline=True,
            linecolor="#E0E0E0",
            mirror=True
        ),
        # 优化交互
        hovermode="closest",
        hoverlabel=dict(
            bgcolor="white",
            font_size=12,
            font_family="Arial",
            bordercolor=PURPLE_COLORS["medium"]
        ),
        # 调整边距
        margin=dict(l=60, r=40, t=100, b=60),
        # 使用更美观的模板
        template="plotly_white",
        # 添加评估指标标注
        annotations=[
            dict(
                x=0.05,
                y=0.95,
                xref="paper",
                yref="paper",
                text=f"R² = {r2:.4f}<br>MAE = {mae:.4f}<br>RMSE = {rmse:.4f}",
                showarrow=False,
                font=dict(
                    size=13, 
                    color=PURPLE_COLORS["darkest"],
                    family="Arial"
                ),
                bgcolor="rgba(255, 255, 255, 0.9)",
                bordercolor=PURPLE_COLORS["dark"],
                borderwidth=1,
                borderpad=12
            )
        ],
        # 添加图例
        legend=dict(
            title="数据类型",
            orientation="h",
            yanchor="top",
            y=0.9,
            xanchor="center",
            x=0.5,
            font=dict(
                size=12,
                color="#666666"
            ),
            bgcolor="rgba(255, 255, 255, 0.8)",
            bordercolor=PURPLE_COLORS["light"],
            borderwidth=1
        )
    )
    
    return fig


# 柱状图 - 预测模型交叉验证结果（带误差线）
def plot_prediction_cv_bar(cv_results_df):
    """
    绘制预测模型交叉验证结果柱状图（带误差线）
    
    参数:
        cv_results_df: DataFrame - 包含交叉验证结果的DataFrame
    
    返回:
        fig: Plotly Figure - 生成的图表
    """
    # 创建子图
    from plotly.subplots import make_subplots
    
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=('MAE（越小越好）', 'RMSE（越小越好）', 'R²（越接近1越好）'),
        horizontal_spacing=0.1
    )
    
    # MAE
    fig.add_trace(go.Bar(
        x=cv_results_df['model'],
        y=cv_results_df['mae_mean'],
        error_y=dict(
            type='data',
            array=cv_results_df['mae_std'],
            visible=True,
            color=PURPLE_COLORS['darker'],
            thickness=1.5,
            width=8
        ),
        marker=dict(color=PURPLE_COLORS['dark'], line=dict(color=PURPLE_COLORS['darker'], width=1), opacity=0.8),
        name='MAE',
        showlegend=False
    ), row=1, col=1)
    
    # RMSE
    fig.add_trace(go.Bar(
        x=cv_results_df['model'],
        y=cv_results_df['rmse_mean'],
        error_y=dict(
            type='data',
            array=cv_results_df['rmse_std'],
            visible=True,
            color=PURPLE_COLORS['darker'],
            thickness=1.5,
            width=8
        ),
        marker=dict(color=PURPLE_COLORS['medium'], line=dict(color=PURPLE_COLORS['darker'], width=1), opacity=0.8),
        name='RMSE',
        showlegend=False
    ), row=1, col=2)
    
    # R²
    fig.add_trace(go.Bar(
        x=cv_results_df['model'],
        y=cv_results_df['r2_mean'],
        error_y=dict(
            type='data',
            array=cv_results_df['r2_std'],
            visible=True,
            color=PURPLE_COLORS['darker'],
            thickness=1.5,
            width=8
        ),
        marker=dict(color=PURPLE_COLORS['light'], line=dict(color=PURPLE_COLORS['darker'], width=1), opacity=0.8),
        name='R²',
        showlegend=False
    ), row=1, col=3)
    
    # 更新布局
    fig.update_layout(
        title=dict(
            text="5折交叉验证预测模型性能（均值±标准差）",
            x=0.5,
            font=dict(size=18, color="#333333")
        ),
        margin=dict(l=60, r=40, t=100, b=100),
        template="plotly_white",
        height=500
    )
    
    # 更新所有子图的x轴
    for i in range(1, 4):
        fig.update_xaxes(tickangle=45, row=1, col=i)
    
    return fig


# 直方图 - PHQ-9总分分布
def plot_phq9_distribution(depression_df):
    """
    绘制PHQ-9总分分布直方图
    
    参数:
        depression_df: DataFrame - 包含抑郁症状数据的DataFrame
    
    返回:
        fig: Plotly Figure - 生成的图表
    """
    # 计算样本数量
    n_samples = len(depression_df)
    
    # 计算PHQ-9总分的频率分布
    score_counts = depression_df['phq9_total'].value_counts().sort_index()
    
    # 创建完整的0-27分的分布
    full_scores = pd.DataFrame({'score': range(0, 28)})
    full_counts = full_scores.merge(score_counts.rename('count'), left_on='score', right_index=True, how='left').fillna(0)
    
    # 定义抑郁程度区间和颜色
    def get_depression_color(score):
        if score <= 4:
            return '#E0E0E0'  # 浅灰色 - 无抑郁
        elif score <= 9:
            return '#B3E5FC'  # 浅蓝色 - 轻度
        elif score <= 14:
            return '#FFF59D'  # 浅黄色 - 中度
        elif score <= 19:
            return '#FFCC80'  # 橙色 - 中重度
        else:
            return '#FFAB91'  # 红色 - 重度
    
    # 应用颜色映射
    full_counts['color'] = full_counts['score'].apply(get_depression_color)
    
    # 创建柱状图
    fig = go.Figure()
    
    # 添加柱子
    fig.add_trace(go.Bar(
        x=full_counts['score'],
        y=full_counts['count'],
        marker=dict(
            color=full_counts['color'],
            line=dict(color="white", width=1)
        ),
        width=0.8  # 柱子宽度
    ))
    
    # 添加频数标签
    for i, row in full_counts.iterrows():
        if row['count'] > 0:
            fig.add_annotation(
                x=row['score'],
                y=row['count'] + 0.1,
                text=str(int(row['count'])),
                font=dict(size=10),
                showarrow=False,
                yanchor='bottom'
            )
    
    # 添加垂直虚线和区间标注
    thresholds = [5, 10, 15, 20]
    labels = ['无抑郁', '轻度', '中度', '中重度', '重度']
    label_positions = [2.5, 7.5, 12.5, 17.5, 23.5]
    
    for threshold in thresholds:
        fig.add_vline(x=threshold - 0.5, line_dash="dash", line_color="gray", line_width=1)
    
    for i, (pos, label) in enumerate(zip(label_positions, labels)):
        fig.add_annotation(
            x=pos,
            y=4.5,
            text=label,
            font=dict(size=10),
            showarrow=False,
            xanchor='center'
        )
    
    # 优化布局
    fig.update_layout(
        title=f"PHQ-9总分分布（N={n_samples}）",
        xaxis=dict(
            title="PHQ-9总分",
            tickmode="linear",
            dtick=1,
            range=[-0.5, 27.5],
            gridcolor="#F0F0F0"
        ),
        yaxis=dict(
            title="样本数量",
            range=[0, 5],
            tickmode="linear",
            dtick=1,
            gridcolor="#F0F0F0"
        ),
        hovermode="x unified",
        hoverlabel=dict(bgcolor="white", font_size=12, font_family="Arial"),
        margin=dict(l=40, r=40, t=60, b=40),
        height=600,
        template="plotly_white"
    )
    
    return fig

# 饼图 - 抑郁严重程度分布
def plot_depression_severity_pie(depression_df):
    """
    绘制抑郁严重程度分布饼图
    
    参数:
        depression_df: DataFrame - 包含抑郁症状数据的DataFrame
    
    返回:
        fig: Plotly Figure - 生成的图表
    """
    severity_counts = depression_df["depression_severity"].value_counts()
    df = pd.DataFrame({
        "严重程度": severity_counts.index,
        "数量": severity_counts.values
    })
    
    # 确保数据按照正常、轻度抑郁、中度抑郁、重度抑郁的顺序排列
    severity_order = ["正常", "轻度抑郁", "中度抑郁", "重度抑郁"]
    df = df.set_index("严重程度").reindex(severity_order).reset_index()
    # 过滤掉可能的NaN值
    df = df.dropna()
    
    # 为每个严重程度指定颜色，确保中度抑郁用深色，正常用浅色
    # 颜色顺序：正常(最浅) < 轻度抑郁 < 中度抑郁(较深) < 重度抑郁(最深)
    color_map = {
        "正常": PURPLE_COLORS["light"],
        "轻度抑郁": PURPLE_COLORS["medium"],
        "中度抑郁": PURPLE_COLORS["dark"],
        "重度抑郁": PURPLE_COLORS["darker"]
    }
    
    # 根据严重程度获取对应的颜色
    colors = [color_map.get(severity, PURPLE_COLORS["light"]) for severity in df["严重程度"]]
    
    fig = px.pie(
        df,
        values="数量",
        names="严重程度",
        title="抑郁严重程度分布",
        color_discrete_sequence=colors,
        template="plotly_white"
    )
    
    # 添加交互功能
    fig.update_traces(
        textinfo="label+percent",
        hoverinfo="label+value+percent",
        # 添加悬停效果
        hovertemplate="<b>%{label}</b><br>数量: %{value}<br>占比: %{percent}<extra></extra>",
        marker=dict(
            line=dict(color="white", width=1)
        ),
        # 添加悬停时的动画效果
        hoverlabel=dict(
            bgcolor="white",
            font_size=14,
            font_family="Arial",
            bordercolor=PURPLE_COLORS["dark"]
        ),

        # 悬停时的动画效果
        opacity=0.8
    )
    
    # 添加饼图扇区悬停时的突出效果
    fig.update_layout(
        hovermode="closest",
        margin=dict(l=40, r=40, t=60, b=40),
        # 添加过渡动画
        transition={"duration": 500}
    )
    
    return fig

# 热力图 - 症状相关性
def plot_symptom_correlation(corr_matrix):
    """
    绘制症状相关性热力图
    
    参数:
        corr_matrix: DataFrame - 相关性矩阵
    
    返回:
        fig: Plotly Figure - 生成的图表
    """
    # 特征名汉化
    symptom_names = {
        'phq1': '兴趣低落',
        'phq2': '情绪低落',
        'phq3': '睡眠问题',
        'phq4': '疲劳/精力差',
        'phq5': '食欲问题',
        'phq6': '自责/无价值感',
        'phq7': '注意力不集中',
        'phq8': '动作/言语问题',
        'phq9': '自杀意念'
    }
    
    # 重命名列和索引
    corr_matrix = corr_matrix.rename(columns=symptom_names, index=symptom_names)
    
    fig = px.imshow(
        corr_matrix,
        title="症状相关性热力图",
        labels=dict(color="相关性", x="症状", y="症状"),
        color_continuous_scale=["#E6E6FA", "#9370DB"],  # 使用#9370DB渐变
        template="plotly_white"
    )
    
    # 添加白色细边框效果
    fig.update_traces(
        xgap=1,  # 水平间隙
        ygap=1,  # 垂直间隙
        colorscale=["#E6E6FA", "#9370DB"]  # 确保颜色渐变正确
    )
    
    # 优化布局
    fig.update_layout(
        hovermode="closest",
        hoverlabel=dict(bgcolor="white", font_size=12, font_family="Arial"),
        margin=dict(l=100, r=40, t=60, b=100),
        height=600,  # 设置图表高度
        # 设置标题颜色
        title=dict(
            text="症状相关性热力图",
            font=dict(color="#333333")  # 深灰色文字
        )
    )
    
    return fig

# 柱状图 - 自杀倾向类别分布
def plot_suicide_class_distribution(suicide_df):
    """
    绘制自杀倾向类别分布柱状图
    
    参数:
        suicide_df: DataFrame - 包含自杀检测数据的DataFrame
    
    返回:
        fig: Plotly Figure - 生成的图表
    """
    fig = px.bar(
        suicide_df,
        x="class",
        title="自杀倾向类别分布",
        labels={"class": "类别", "count": "数量"},
        color_discrete_sequence=[PURPLE_COLORS["medium"]],
        template="plotly_white"
    )
    
    # 添加圆角
    fig.update_traces(marker=dict(line=dict(color=PURPLE_COLORS["darker"], width=1), opacity=0.8), selector=dict(type="bar"))
    
    # 添加交互功能
    fig.update_layout(
        hovermode="x unified",
        hoverlabel=dict(bgcolor="white", font_size=12, font_family="Arial"),
        margin=dict(l=40, r=40, t=60, b=40)
    )
    
    return fig

# 折线图 - 肘部法则
def plot_elbow_method(k_values, inertia):
    """
    绘制肘部法则折线图
    
    参数:
        k_values: list - 簇数列表
        inertia: list - 对应的惯性值
    
    返回:
        fig: Plotly Figure - 生成的图表
    """
    df = pd.DataFrame({
        "簇数": k_values,
        "惯性值": inertia
    })
    
    # 找到拐点（简单方法：寻找惯性值下降速率变化最大的点）
    elbow_point = None
    if len(inertia) > 2:
        # 计算相邻点之间的惯性值差值
        differences = [inertia[i] - inertia[i+1] for i in range(len(inertia)-1)]
        # 计算差值的变化率
        if len(differences) > 1:
            change_rates = [differences[i] - differences[i+1] for i in range(len(differences)-1)]
            # 找到变化率最大的点
            max_change_idx = change_rates.index(max(change_rates))
            elbow_point = k_values[max_change_idx + 1]
    
    fig = px.line(
        df,
        x="簇数",
        y="惯性值",
        title="肘部法确定最优聚类数",
        labels={"簇数": "聚类数 K", "惯性值": "SSE/Inertia"},
        markers=True,
        template="plotly_white"
    )
    
    # 设置线条和标记样式
    fig.update_traces(
        line=dict(color=PURPLE_COLORS["dark"], width=2),
        marker=dict(color=PURPLE_COLORS["darkest"], size=8)
    )
    
    # 添加拐点标注
    if elbow_point:
        # 找到拐点对应的惯性值
        elbow_inertia = inertia[k_values.index(elbow_point)]
        # 添加垂直虚线
        fig.add_vline(
            x=elbow_point,
            line_dash="dash",
            line_color="#9370DB",
            line_width=1.5
        )
        # 添加标注
        fig.add_annotation(
            x=elbow_point,
            y=elbow_inertia,
            text=f"最优K={elbow_point}",
            showarrow=True,
            arrowhead=2,
            ax=50,
            ay=-30,
            font=dict(size=12, color="#333333"),
            bgcolor="white",
            bordercolor="#9370DB",
            borderwidth=1
        )
    
    # 优化布局
    fig.update_layout(
        hovermode="x unified",
        hoverlabel=dict(bgcolor="white", font_size=12, font_family="Arial"),
        margin=dict(l=60, r=40, t=80, b=60),
        # 标题居中
        title=dict(
            text="肘部法确定最优聚类数",
            x=0.5,
            y=0.95,
            font=dict(
                size=18,
                color="#333333"
            )
        ),
        # 坐标轴设置
        xaxis=dict(
            title=dict(text="聚类数 K", font=dict(size=14, color="#333333")),
            tickfont=dict(size=12),
            showgrid=False,
            tickmode="linear",
            tick0=1,
            dtick=1
        ),
        yaxis=dict(
            title=dict(text="SSE/Inertia", font=dict(size=14, color="#333333")),
            tickfont=dict(size=12),
            showgrid=False
        ),
        # 白色背景
        plot_bgcolor="white",
        paper_bgcolor="white"
    )
    
    return fig

# 词云图 - 自杀倾向文本分析
def generate_wordcloud(text_data, output_path="outputs/images/suicide_wordcloud.png"):
    """生成词云图，支持中文"""
    if not text_data or not text_data.strip():
        raise ValueError("词云文本数据为空")

    # 中文字体路径（Windows 微软雅黑）
    font_path = "C:/Windows/Fonts/msyh.ttc"

    wordcloud = WordCloud(
        width=800,
        height=400,
        background_color="white",
        colormap="Purples",
        max_words=200,
        max_font_size=100,
        font_path=font_path,
        random_state=42
    )

    wordcloud.generate(text_data)
    wordcloud.to_file(output_path)
    return output_path

# 折线图 - 时间序列趋势
def plot_time_series_trend(depression_df, user_id=None):
    """
    绘制抑郁症状随时间的变化趋势
    
    参数:
        depression_df: DataFrame - 包含抑郁症状数据的DataFrame
        user_id: int - 可选，指定用户ID
    
    返回:
        fig: Plotly Figure - 生成的图表
    """
    # 如果指定了用户ID，只选择该用户的数据
    if user_id:
        df = depression_df[depression_df['user_id'] == user_id].copy()
        title = f"用户 {user_id} 的抑郁症状时间趋势"
        subtitle = f"用户 {user_id} 抑郁症状时间趋势（含30天移动平均）"
    else:
        # 否则使用所有用户的平均数据
        df = depression_df.copy()
        title = "抑郁症状时间趋势"
        subtitle = "所有用户抑郁症状平均时间趋势（含30天移动平均）"
        # 按时间分组计算平均值
        df = df.groupby('phq.day').agg({'phq9_total': 'mean'}).reset_index()
    
    # 按时间排序
    df = df.sort_values('phq.day')
    
    # 计算30天移动平均
    df['30_day_ma'] = df['phq9_total'].rolling(window=30, min_periods=1).mean()
    
    # 创建Figure对象
    fig = go.Figure()
    
    # 添加原始数据曲线
    fig.add_trace(go.Scatter(
        x=df["phq.day"],
        y=df["phq9_total"],
        name="原始数据",
        line=dict(
            color="#9370DB",  # 淡紫色
            width=1,  # 更细的线
            # 平滑曲线
            shape="spline",
            smoothing=1.3
        ),
        opacity=0.3,  # 透明度30%
        mode="lines"
    ))
    
    # 添加30天移动平均曲线
    fig.add_trace(go.Scatter(
        x=df["phq.day"],
        y=df["30_day_ma"],
        name="30天移动平均",
        line=dict(
            color="#999999",  # 稍深的灰色，更突出
            width=2.5,  # 更粗的线
            # 平滑曲线
            shape="spline",
            smoothing=1.3
        ),
        mode="lines"
    ))
    
    # 添加参考线（使用浅灰色虚线）
    fig.add_hline(y=5, line_dash="dash", line_color="#E0E0E0", name="轻度抑郁阈值(5分)")
    fig.add_hline(y=10, line_dash="dash", line_color="#C0C0C0", name="中度抑郁阈值(10分)")
    fig.add_hline(y=15, line_dash="dash", line_color="#A0A0A0", name="重度抑郁阈值(15分)")
    
    # 优化布局和交互功能
    fig.update_layout(
        hovermode="x",  # 鼠标悬停显示单个点的信息
        hoverlabel=dict(bgcolor="white", font_size=12, font_family="Arial"),
        margin=dict(l=40, r=40, t=100, b=40),
        height=600,  # 设置图表高度
        # 美化标题
        title=dict(
            text=f"{title}<br><sub>{subtitle}</sub>",
            x=0.5,  # 居中
            y=0.95,
            font=dict(
                size=18,  # 字号稍放大
                color="#333333"
            )
        ),
        # 添加图例
        legend=dict(
            title="数据类型",
            orientation="v",
            yanchor="top",
            y=1.0,
            xanchor="right",
            x=1.0,
            font=dict(
                size=12
            )
        ),
        # 启用缩放、平移等交互功能
        xaxis=dict(
            title=dict(
                text="时间（天）",
                font=dict(
                    size=14,  # 字号稍放大
                    color="#666666"  # 深灰色
                )
            ),
            # 移除范围滑块
            type="linear"
        ),
        yaxis=dict(
            title=dict(
                text="PHQ-9总分",
                font=dict(
                    size=14,  # 字号稍放大
                    color="#666666"  # 深灰色
                )
            ),
            fixedrange=False  # 允许y轴缩放
        ),
        template="plotly_white"
    )
    
    # 自定义悬停提示
    fig.update_traces(
        hovertemplate="时间（天）: %{x}<br>PHQ-9总分: %{y:.2f}<extra></extra>"
    )
    
    return fig

# 柱状图 - 人口统计学分析
def plot_demographic_analysis(depression_df):
    """
    绘制人口统计学分析图表
    
    参数:
        depression_df: DataFrame - 包含抑郁症状数据的DataFrame
    
    返回:
        fig: Plotly Figure - 生成的图表
    """
    # 按性别分组计算平均PHQ-9分数，并将英文标签映射为中文
    sex_cn_map = {'female': '女性', 'male': '男性', 'transgender': '跨性别'}
    gender_df = depression_df.groupby('sex').agg({'phq9_total': 'mean'}).reset_index()
    gender_df['性别'] = gender_df['sex'].map(sex_cn_map).fillna(gender_df['sex'])

    fig = px.bar(
        gender_df,
        x='性别',
        y='phq9_total',
        title='不同性别的平均PHQ-9分数',
        labels={'phq9_total': '平均PHQ-9分数'},
        color="phq9_total",
        color_continuous_scale=[PURPLE_COLORS["light"], PURPLE_COLORS["dark"]],
        template="plotly_white"
    )
    
    # 添加圆角
    fig.update_traces(marker=dict(line=dict(color=PURPLE_COLORS["darker"], width=1), opacity=0.8), selector=dict(type="bar"))
    
    # 添加交互功能
    fig.update_layout(
        hovermode="x unified",
        hoverlabel=dict(bgcolor="white", font_size=12, font_family="Arial"),
        margin=dict(l=40, r=40, t=60, b=40)
    )
    
    return fig

# 散点图 - 幸福感与抑郁关系
def plot_happiness_depression_relation(depression_df):
    """
    绘制幸福感与抑郁关系散点图
    
    参数:
        depression_df: DataFrame - 包含抑郁症状数据的DataFrame
    
    返回:
        fig: Plotly Figure - 生成的图表
    """
    # 确保数据中包含所需列
    if "happiness.score" not in depression_df.columns or "phq9_total" not in depression_df.columns:
        # 创建一个空图表并显示错误信息
        fig = go.Figure()
        fig.update_layout(
            title="幸福感与抑郁程度的关系",
            xaxis=dict(title="幸福感评分"),
            yaxis=dict(title="PHQ-9总分"),
            annotations=[
                dict(
                    x=0.5,
                    y=0.5,
                    xref="paper",
                    yref="paper",
                    text="数据中缺少幸福感评分或PHQ-9总分列",
                    showarrow=False,
                    font=dict(size=14, color="red")
                )
            ],
            margin=dict(l=40, r=40, t=60, b=40)
        )
        return fig
    
    # 使用Plotly Express创建散点图
    fig = px.scatter(
        depression_df,
        x="happiness.score",
        y="phq9_total",
        title="幸福感与抑郁程度的关系",
        labels={"happiness.score": "幸福感评分", "phq9_total": "PHQ-9总分"},
        color="phq9_total",
        color_continuous_scale=[PURPLE_COLORS["light"], PURPLE_COLORS["dark"]],
        template="plotly_white"
    )
    
    # 添加交互功能
    fig.update_traces(marker=dict(opacity=0.7), selector=dict(type="scatter"))
    fig.update_layout(
        hovermode="closest",
        hoverlabel=dict(bgcolor="white", font_size=12, font_family="Arial"),
        margin=dict(l=40, r=40, t=60, b=40)
    )
    
    return fig

# 热力图 - 症状组合分析
def plot_symptom_combination(depression_df):
    """
    绘制症状组合分析热力图
    
    参数:
        depression_df: DataFrame - 包含抑郁症状数据的DataFrame
    
    返回:
        fig: Plotly Figure - 生成的图表
    """
    # 选择PHQ-9的9个症状
    phq_columns = [f"phq{i}" for i in range(1, 10)]
    
    # 计算症状之间的共现频率
    # 首先将症状转换为二分类（0表示无症状，1表示有症状）
    binary_symptoms = depression_df[phq_columns].map(lambda x: 1 if x > 0 else 0)
    
    # 计算共现矩阵
    co_occurrence = binary_symptoms.T.dot(binary_symptoms)
    
    # 标准化共现矩阵（除以总样本数）
    co_occurrence_norm = co_occurrence / len(binary_symptoms)
    
    # 特征名汉化
    symptom_names = {
        'phq1': '兴趣低落',
        'phq2': '情绪低落',
        'phq3': '睡眠问题',
        'phq4': '疲劳/精力差',
        'phq5': '食欲问题',
        'phq6': '自责/无价值感',
        'phq7': '注意力不集中',
        'phq8': '动作/言语问题',
        'phq9': '自杀意念'
    }
    
    # 重命名列和索引
    co_occurrence_norm = co_occurrence_norm.rename(columns=symptom_names, index=symptom_names)
    
    fig = px.imshow(
        co_occurrence_norm,
        title="症状组合分析",
        labels=dict(color="共现频率", x="症状", y="症状"),
        color_continuous_scale=["#E6E6FA", "#9370DB"],  # 使用#9370DB渐变
        template="plotly_white"
    )
    
    # 添加白色细边框效果
    fig.update_traces(
        xgap=1,  # 水平间隙
        ygap=1,  # 垂直间隙
        colorscale=["#E6E6FA", "#9370DB"]  # 确保颜色渐变正确
    )
    
    # 优化布局
    fig.update_layout(
        hovermode="closest",
        hoverlabel=dict(bgcolor="white", font_size=12, font_family="Arial"),
        margin=dict(l=100, r=40, t=60, b=100),
        height=600,  # 设置图表高度
        # 设置标题颜色
        title=dict(
            text="症状组合分析",
            font=dict(color="#333333")  # 深灰色文字
        )
    )

    return fig


# ═══════════════════════════════════════════════════════
# TF-IDF 特征重要性按词类聚合
# ═══════════════════════════════════════════════════════

# 六大词类关键词表（小写）
_CATEGORY_KEYWORDS = {
    '死亡相关词': [
        'die', 'kill', 'end', 'death', 'suicide', 'dead', 'dying',
        'suicidal', 'murder', 'grave', 'funeral', 'bury',
        'corpse', 'hanging', 'hang', 'overdose', 'cut', 'bleed',
    ],
    '否定词': [
        'dont', 'cant', 'not', 'never', 'nothing', 'no', 'neither',
        'nor', 'without', 'none', 'hardly', 'barely', 'wont', 'wouldnt',
        'couldnt', 'shouldnt', 'doesnt', 'didnt', 'hasnt', 'hadnt',
        'isnt', 'arent', 'wasnt',
    ],
    '第一人称代词': [
        'i', 'my', 'me', 'myself', 'mine', 'im', 'ive', 'ill', 'id',
        'we', 'our', 'us', 'ourselves',
    ],
    '情感/痛苦词': [
        'feel', 'pain', 'sad', 'hate', 'hurt', 'alone', 'lonely',
        'hopeless', 'worthless', 'cry', 'afraid', 'scared', 'depressed',
        'anxiety', 'anxious', 'suffering', 'misery', 'empty', 'numb',
        'tired', 'exhausted', 'angry', 'rage', 'guilt', 'shame',
        'despair', 'broken', 'lost', 'dark', 'sorrow',
    ],
    '统计特征': [
        '文本长度', '自杀关键词数量', '第一人称占比',
        '情感得分', '消极词占比', '情绪强度',
    ],
}


def aggregate_tfidf_importance(importance_df):
    """
    将 TF-IDF 特征重要性按词类聚合。

    参数:
        importance_df: DataFrame — 含 feature / importance 两列

    返回:
        aggregated: DataFrame — 含「类别」「重要性」两列，按重要性降序排列
    """
    # 统计特征名映射
    stat_name_map = {
        'text_length_scaled': '文本长度',
        'suicide_keyword_count_scaled': '自杀关键词数量',
        'first_person_ratio_scaled': '第一人称占比',
        'sentiment_score_scaled': '情感得分',
        'negative_word_ratio_scaled': '消极词占比',
        'emotion_intensity_scaled': '情绪强度',
    }

    # 构建 词 → 类别 的反向索引
    word_to_cat = {}
    for cat, words in _CATEGORY_KEYWORDS.items():
        for w in words:
            word_to_cat[w] = cat

    stat_keywords = set(_CATEGORY_KEYWORDS['统计特征'])

    cat_sums = {cat: 0.0 for cat in _CATEGORY_KEYWORDS}
    cat_sums['其他词'] = 0.0

    for _, row in importance_df.iterrows():
        feat = str(row['feature'])
        imp = float(row['importance'])

        # 1) 匹配统计特征（中文名或英文列名）
        if feat in stat_keywords or stat_name_map.get(feat, '') in stat_keywords:
            cat_sums['统计特征'] += imp
            continue

        # 2) 去掉 tfidf_ 前缀
        token = feat
        if token.startswith('tfidf_'):
            token = token[6:]

        # 对 bigram 拆分后匹配每个词
        parts = token.lower().split('_') if '_' in token else [token.lower()]

        matched = False
        for part in parts:
            if part in word_to_cat:
                cat_sums[word_to_cat[part]] += imp
                matched = True
                break

        if not matched:
            cat_sums['其他词'] += imp

    # 构建结果
    records = [
        {'类别': cat, '重要性': round(val, 6)}
        for cat, val in cat_sums.items()
        if val > 0
    ]
    aggregated = pd.DataFrame(records).sort_values('重要性', ascending=False).reset_index(drop=True)
    return aggregated


def plot_category_importance(aggregated_df):
    """
    绘制按词类聚合的特征重要性水平柱状图（plotly）。

    参数:
        aggregated_df: DataFrame — 含「类别」「重要性」两列

    返回:
        fig: Plotly Figure
    """
    df = aggregated_df.sort_values('重要性', ascending=True)

    fig = px.bar(
        df,
        x='重要性',
        y='类别',
        orientation='h',
        title='特征重要性（按词类聚合）',
        labels={'重要性': '类别总重要性', '类别': '词类'},
        color_discrete_sequence=['#9370DB'],
        template='plotly_white',
    )

    fig.update_traces(
        marker=dict(
            line=dict(color='#7B68EE', width=1),
            opacity=0.8,
        ),
    )

    fig.update_layout(
        hovermode='y',
        hoverlabel=dict(bgcolor='white', font_size=12),
        yaxis=dict(categoryorder='total ascending'),
        height=max(300, 60 * len(df)),
        margin=dict(l=20, r=20, t=50, b=20),
    )

    return fig
