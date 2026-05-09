"""生成系统架构图，用于毕业论文答辩 — 纯 matplotlib 实现"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
import os


def create_architecture_diagram():
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
    plt.rcParams["axes.unicode_minus"] = False

    fig, ax = plt.subplots(1, 1, figsize=(18, 13))
    ax.set_xlim(0, 18)
    ax.set_ylim(0, 13)
    ax.axis("off")
    ax.set_facecolor("#F8F8FF")
    fig.patch.set_facecolor("#F8F8FF")

    COLORS = {
        "data":   "#D6EAF8",  # 蓝
        "prep":   "#FCF3CF",  # 黄
        "model":  "#D5F5E3",  # 绿
        "vis":    "#E8DAEF",  # 紫
        "web":    "#FADBD8",  # 粉
        "output": "#E5E7E9",  # 灰
    }

    def draw_box(ax, x, y, w, h, color, title, lines, fontsize=10):
        """绘制圆角矩形模块"""
        box = FancyBboxPatch(
            (x - w / 2, y - h / 2), w, h,
            boxstyle="round,pad=0.15", facecolor=color,
            edgecolor="#999999", linewidth=1.2, alpha=0.95
        )
        ax.add_patch(box)
        ax.text(x, y + h / 2 - 0.25, title, ha="center", va="top",
                fontsize=fontsize + 2, fontweight="bold", color="#333333")
        for i, line in enumerate(lines):
            ax.text(x, y + h / 2 - 0.6 - i * 0.32, line, ha="center", va="top",
                    fontsize=fontsize, color="#555555")

    def draw_arrow(x1, y1, x2, y2, label="", style="solid", color="#666666"):
        """绘制连接箭头"""
        ax.annotate(
            "", xy=(x2, y2), xytext=(x1, y1),
            arrowprops=dict(
                arrowstyle="->", color=color, lw=1.5,
                linestyle=style, connectionstyle="arc3,rad=0"
            )
        )
        if label:
            mx, my = (x1 + x2) / 2, (y1 + y2) / 2
            ax.text(mx + 0.15, my, label, fontsize=8, color="#555555",
                    ha="center", va="bottom", style="italic")

    # ══════════════════════════════════════════
    # 标题
    # ══════════════════════════════════════════
    ax.text(9, 12.6, "抑郁症多维度心理特征分析系统 — 系统架构图",
            ha="center", fontsize=18, fontweight="bold", color="#333333")

    # ══════════════════════════════════════════
    # Layer labels
    # ══════════════════════════════════════════
    layers = [
        (0.8,  "数据层",  "#2E86C1"),
        (1.0,  "预处理层", "#D4AC0D"),
        (7.0,  "模型层",   "#27AE60"),
        (13.0, "可视化层", "#8E44AD"),
        (14.0, "前端层",   "#E74C3C"),
        (10.0, "输出层",   "#7F8C8D"),
    ]

    # ══════════════════════════════════════════
    # Row 1: 数据层 (y=11)
    # ══════════════════════════════════════════
    draw_box(ax, 4, 11, 3.5, 1.2, COLORS["data"],
             "PHQ-9 抑郁症状数据集",
             ["14天追踪 · 症状评分 · 人口统计"])
    draw_box(ax, 13, 11, 3.5, 1.2, COLORS["data"],
             "Suicide Detection 数据集",
             ["自杀/非自杀标签 · 文本内容"])

    # ══════════════════════════════════════════
    # Row 2: 预处理层 (y=9)
    # ══════════════════════════════════════════
    draw_box(ax, 4, 9, 3.5, 1.6, COLORS["prep"],
             "数据预处理 (depression)",
             ["缺失值分组均值填充 · 异常值校验",
              "PHQ-9 总分/核心症状/症状计数",
              "StandardScaler 标准化"])
    draw_box(ax, 13, 9, 3.5, 1.6, COLORS["prep"],
             "数据预处理 (suicide)",
             ["编码检测 · jieba/nltk 分词",
              "关键词计数 · 情感极性 · BERT",
              "SMOTE 过采样平衡"])

    # ══════════════════════════════════════════
    # Row 3: 模型层 (y=6.3)
    # ══════════════════════════════════════════
    draw_box(ax, 3, 6.3, 3.2, 2.0, COLORS["model"],
             "聚类分析 (K-Means)",
             ["PHQ-9 9项症状特征",
              "肘部法则 → K=3",
              "t-SNE 2D 可视化",
              "簇画像统计"])
    draw_box(ax, 9, 6.3, 3.2, 2.0, COLORS["model"],
             "分类模型 (自杀识别)",
             ["LR / RF / XGBoost",
              "SMOTE + 5折分层CV",
              "F1 / AUC 评估",
              "SHAP 特征重要性"])
    draw_box(ax, 15, 6.3, 3.2, 2.0, COLORS["model"],
             "回归预测 (PHQ-9)",
             ["RF / XGBoost / Stacking",
              "5折KFold CV",
              "R² / MAE / RMSE",
              "残差分布分析"])

    # ══════════════════════════════════════════
    # Row 4: 可视化 + 前端 (y=3.5)
    # ══════════════════════════════════════════
    draw_box(ax, 5.5, 3.7, 4.5, 1.8, COLORS["vis"],
             "可视化引擎 (visualization.py)",
             ["Plotly 交互图 (滚轮缩放)",
              "词云 · 相关性热力图 · 时间趋势",
              "柱状图 (误差线) · 饼图 · 散点图"])

    draw_box(ax, 13, 3.7, 5.5, 1.8, COLORS["web"],
             "Streamlit Web 前端 (app.py + main.py)",
             ["用户登录/注册 · 四模块导航切换",
              "session_state 状态管理 · 模型重建按钮",
              "聚类 | 分类 | 预测 | 数据可视化"])

    # ══════════════════════════════════════════
    # Row 5: 输出层 (y=1.8)
    # ══════════════════════════════════════════
    draw_box(ax, 9, 1.8, 10, 1.2, COLORS["output"],
             "结果输出",
             ["CSV: 簇画像 / F1&AUC / R²&MAE / 特征重要性  |  图片: t-SNE / 混淆矩阵 / 残差 / 词云  |  Plotly 交互图表"])

    # ══════════════════════════════════════════
    # 连接箭头
    # ══════════════════════════════════════════
    # 数据 → 预处理
    draw_arrow(4, 10.4, 4, 9.8)
    draw_arrow(13, 10.4, 13, 9.8)

    # 预处理 → 模型
    draw_arrow(4, 8.2, 3, 7.3)
    draw_arrow(4, 8.2, 9, 7.3)
    draw_arrow(13, 8.2, 15, 7.3)
    draw_arrow(13, 8.2, 9, 7.3)

    # 模型 → 可视化
    draw_arrow(3, 5.3, 5.5, 4.6, style="dashed")
    draw_arrow(9, 5.3, 5.5, 4.6, style="dashed")
    draw_arrow(15, 5.3, 5.5, 4.6, style="dashed")

    # 可视化 → 前端
    draw_arrow(7.75, 3.7, 10.25, 3.7)

    # 前端 → 模型 (触发)
    ax.annotate("", xy=(10, 5.3), xytext=(11.5, 3.7),
                arrowprops=dict(arrowstyle="->", color="#E74C3C", lw=1.2,
                               linestyle="dotted", connectionstyle="arc3,rad=-0.3"))
    ax.text(11.3, 4.5, "重建按钮", fontsize=8, color="#E74C3C", rotation=55)

    # 模型 → 输出
    draw_arrow(3, 5.3, 6, 2.4, style="dotted")
    draw_arrow(9, 5.3, 9, 2.4, style="dotted")
    draw_arrow(15, 5.3, 12, 2.4, style="dotted")

    # 前端 → 输出
    draw_arrow(13, 2.8, 9, 2.4, style="dotted")

    # ══════════════════════════════════════════
    # 图例
    # ══════════════════════════════════════════
    legend_items = [
        ("数据层", COLORS["data"]),
        ("预处理层", COLORS["prep"]),
        ("模型层", COLORS["model"]),
        ("可视化层", COLORS["vis"]),
        ("前端层", COLORS["web"]),
    ]
    patches = [mpatches.Patch(color=c, label=n) for n, c in legend_items]
    leg = ax.legend(handles=patches, loc="lower left", fontsize=9,
                    ncol=5, framealpha=0.9, edgecolor="#CCCCCC")
    leg.set_bbox_to_anchor((0.05, 0.005))

    plt.tight_layout(pad=0.5)
    os.makedirs("outputs/images", exist_ok=True)
    fig.savefig("outputs/images/system_architecture.png", dpi=200,
                bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print("架构图已生成: outputs/images/system_architecture.png")


if __name__ == "__main__":
    create_architecture_diagram()
