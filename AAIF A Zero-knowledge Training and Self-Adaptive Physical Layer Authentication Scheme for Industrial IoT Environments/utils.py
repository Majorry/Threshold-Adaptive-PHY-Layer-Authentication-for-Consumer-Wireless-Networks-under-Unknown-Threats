# -*- coding: gbk -*-
import numpy as np
import torch
import torch.nn as nn
import os
import time
import io
from contextlib import redirect_stdout
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
import scipy.stats as stats
from scipy.stats import ks_2samp
from model import AdaptiveIsolationForest, IsolationForest
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA


def plot_tsne(data, labels, save_path, perplexity=30, max_samples=2000, legend_fontsize=10, figsize=(5, 4), point_size=25):
    """
    绘制 t-SNE 分布图 (紧凑版 PDF)
    :param figsize: [新增] 画布大小，默认 (5, 4) 比较小巧
    :param point_size: [新增] 点的大小，默认 25，画布变小后点要相对变大才不显得空
    :param legend_fontsize: 图例字体大小
    :param perplexity: 困惑度
    """
    # 自动更名保存为 PDF
    base_name_log = os.path.splitext(os.path.basename(save_path))[0]
    print(f"正在生成 t-SNE PDF (perp={perplexity}, size={figsize}) -> {base_name_log}.pdf ...")

    # 1. 展平
    if len(data.shape) > 2:
        data = data.reshape(data.shape[0], -1)

    # 2. 采样
    n_samples = data.shape[0]
    if n_samples > max_samples:
        indices = np.random.choice(n_samples, max_samples, replace=False)
        data = data[indices]
        labels = labels[indices]

    # 3. PCA
    if data.shape[1] > 50:
        pca = PCA(n_components=50)
        data = pca.fit_transform(data)

    # 4. t-SNE
    tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=1000, random_state=42, init='pca', learning_rate='auto')
    tsne_results = tsne.fit_transform(data)

    # 5. 绘图
    # 使用传入的 figsize 控制整体尺寸
    plt.figure(figsize=figsize)

    # 绘制样本
    # s=point_size 控制点的大小，alpha=0.7 让点稍微实一点
    plt.scatter(tsne_results[labels == 1, 0], tsne_results[labels == 1, 1],
                c='blue', label='Normal (Legal)', alpha=0.7, s=point_size)

    plt.scatter(tsne_results[labels == -1, 0], tsne_results[labels == -1, 1],
                c='red', label='Abnormal (Illegal)', alpha=0.7, s=point_size)

    # 去除刻度
    plt.xticks([])
    plt.yticks([])

    # 图例
    # markerscale=1.5 让图例里的点比图上的点稍微大一点点即可，不用大太多
    plt.legend(loc='upper right', frameon=True, fontsize=legend_fontsize, markerscale=1.5)

    # 保存
    save_dir = os.path.dirname(save_path)
    os.makedirs(save_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(save_path))[0]
    pdf_save_path = os.path.join(save_dir, base_name + ".pdf")

    # bbox_inches='tight' 会自动裁掉多余白边，配合小的 figsize，图会非常紧凑
    plt.savefig(pdf_save_path, format='pdf', bbox_inches='tight', pad_inches=0.05)
    plt.close()
    print(f"PDF图像已保存至: {pdf_save_path}")

# =========================================
# FLOPs/参数量计算
# =========================================
def count_conv1d_flops(in_channels, out_channels, kernel_size, input_length, stride=1, padding=0):
    """计算Conv1d层的FLOPs"""
    output_length = (input_length + 2 * padding - kernel_size) // stride + 1
    flops = output_length * out_channels * (kernel_size * in_channels * 2)
    return flops, output_length

def count_convtranspose1d_flops(in_channels, out_channels, kernel_size, input_length, stride=1, padding=0, output_padding=0):
    """计算ConvTranspose1d层的FLOPs"""
    output_length = (input_length - 1) * stride - 2 * padding + kernel_size + output_padding
    flops = output_length * out_channels * (kernel_size * in_channels * 2)
    return flops, output_length

def count_batchnorm1d_flops(num_features, input_length):
    """计算BatchNorm1d层的FLOPs"""
    flops = input_length * num_features * 4
    return flops

def count_activation_flops(input_length, num_features):
    """计算激活函数的FLOPs (LeakyReLU, Tanh等)"""
    flops = input_length * num_features
    return flops

def calculate_autoencoder_flops(input_channels=2, input_length=8188):
    """计算AutoEncoder的总FLOPs"""
    total_flops = 0
    current_length = input_length

    print("=" * 60)
    print("AutoEncoder FLOPs 计算详情")
    print("=" * 60)

    # ===== Encoder =====
    print("\n【Encoder】")
    # Layer 1: Conv1d(2, 16, 3, stride=2, padding=1)
    flops, current_length = count_conv1d_flops(2, 16, 3, current_length, stride=2, padding=1)
    total_flops += flops
    print(f"Conv1d(2→16): {flops:,} FLOPs, 输出长度: {current_length}")

    flops = count_batchnorm1d_flops(16, current_length)
    total_flops += flops
    print(f"BatchNorm1d(16): {flops:,} FLOPs")

    flops = count_activation_flops(current_length, 16)
    total_flops += flops
    print(f"LeakyReLU: {flops:,} FLOPs")

    # Layer 2: Conv1d(16, 32, 3, stride=2, padding=1)
    flops, current_length = count_conv1d_flops(16, 32, 3, current_length, stride=2, padding=1)
    total_flops += flops
    print(f"Conv1d(16→32): {flops:,} FLOPs, 输出长度: {current_length}")

    flops = count_batchnorm1d_flops(32, current_length)
    total_flops += flops
    print(f"BatchNorm1d(32): {flops:,} FLOPs")

    flops = count_activation_flops(current_length, 32)
    total_flops += flops
    print(f"LeakyReLU: {flops:,} FLOPs")

    # Layer 3: Conv1d(32, 64, 3, stride=2, padding=1)
    flops, current_length = count_conv1d_flops(32, 64, 3, current_length, stride=2, padding=1)
    total_flops += flops
    print(f"Conv1d(32→64): {flops:,} FLOPs, 输出长度: {current_length}")

    flops = count_batchnorm1d_flops(64, current_length)
    total_flops += flops
    print(f"BatchNorm1d(64): {flops:,} FLOPs")

    flops = count_activation_flops(current_length, 64)
    total_flops += flops
    print(f"LeakyReLU: {flops:,} FLOPs")

    # Layer 4: Conv1d(64, 128, 3, stride=2, padding=1)
    flops, current_length = count_conv1d_flops(64, 128, 3, current_length, stride=2, padding=1)
    total_flops += flops
    print(f"Conv1d(64→128): {flops:,} FLOPs, 输出长度: {current_length}")

    flops = count_batchnorm1d_flops(128, current_length)
    total_flops += flops
    print(f"BatchNorm1d(128): {flops:,} FLOPs")

    flops = count_activation_flops(current_length, 128)
    total_flops += flops
    print(f"LeakyReLU: {flops:,} FLOPs")

    # ===== Decoder =====
    print("\n【Decoder】")
    # Layer 1: ConvTranspose1d(128, 64, 3, stride=2, padding=1, output_padding=1)
    flops, current_length = count_convtranspose1d_flops(128, 64, 3, current_length, stride=2, padding=1,
                                                        output_padding=1)
    total_flops += flops
    print(f"ConvTranspose1d(128→64): {flops:,} FLOPs, 输出长度: {current_length}")

    flops = count_batchnorm1d_flops(64, current_length)
    total_flops += flops
    print(f"BatchNorm1d(64): {flops:,} FLOPs")

    flops = count_activation_flops(current_length, 64)
    total_flops += flops
    print(f"LeakyReLU: {flops:,} FLOPs")

    # Layer 2: ConvTranspose1d(64, 32, 3, stride=2, padding=1, output_padding=0)
    flops, current_length = count_convtranspose1d_flops(64, 32, 3, current_length, stride=2, padding=1,
                                                        output_padding=0)
    total_flops += flops
    print(f"ConvTranspose1d(64→32): {flops:,} FLOPs, 输出长度: {current_length}")

    flops = count_batchnorm1d_flops(32, current_length)
    total_flops += flops
    print(f"BatchNorm1d(32): {flops:,} FLOPs")

    flops = count_activation_flops(current_length, 32)
    total_flops += flops
    print(f"LeakyReLU: {flops:,} FLOPs")

    # Layer 3: ConvTranspose1d(32, 16, 3, stride=2, padding=1, output_padding=1)
    flops, current_length = count_convtranspose1d_flops(32, 16, 3, current_length, stride=2, padding=1,
                                                        output_padding=1)
    total_flops += flops
    print(f"ConvTranspose1d(32→16): {flops:,} FLOPs, 输出长度: {current_length}")

    flops = count_batchnorm1d_flops(16, current_length)
    total_flops += flops
    print(f"BatchNorm1d(16): {flops:,} FLOPs")

    flops = count_activation_flops(current_length, 16)
    total_flops += flops
    print(f"LeakyReLU: {flops:,} FLOPs")

    # Layer 4: ConvTranspose1d(16, 2, 3, stride=2, padding=1, output_padding=1)
    flops, current_length = count_convtranspose1d_flops(16, input_channels, 3, current_length, stride=2, padding=1,
                                                        output_padding=1)
    total_flops += flops
    print(f"ConvTranspose1d(16→{input_channels}): {flops:,} FLOPs, 输出长度: {current_length}")

    flops = count_activation_flops(current_length, input_channels)
    total_flops += flops
    print(f"Tanh: {flops:,} FLOPs")

    print("\n" + "=" * 60)
    print(f"总FLOPs: {total_flops:,} ({total_flops / 1e6:.2f}M)")
    print(f"总GFLOPs: {total_flops / 1e9:.4f}")
    print("=" * 60)

    return total_flops

def count_model_parameters(model):
    """统计模型的参数量"""
    total_params = 0
    trainable_params = 0

    print("\n" + "=" * 60)
    print("模型参数统计")
    print("=" * 60)

    for name, param in model.named_parameters():
        num_params = param.numel()
        total_params += num_params
        if param.requires_grad:
            trainable_params += num_params

        # 分离权重和偏置
        if 'weight' in name:
            param_type = '权重'
        elif 'bias' in name:
            param_type = '偏置'
        else:
            param_type = '其他'

        print(f"{name:50s} | {param_type:6s} | 形状: {str(list(param.shape)):20s} | 参数量: {num_params:>10,}")

    print("=" * 60)
    print(f"总参数量: {total_params:,} ({total_params / 1e6:.2f}M)")
    print(f"可训练参数量: {trainable_params:,} ({trainable_params / 1e6:.2f}M)")
    print(f"不可训练参数量: {total_params - trainable_params:,}")
    print("=" * 60)

    return {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'non_trainable_params': total_params - trainable_params
    }

def print_model_complexity(model, input_channels=2, input_length=8188):
    """打印模型的FLOPs和参数统计信息"""
    print("\n" + "*" * 30)
    print("模型复杂度分析")
    print("*" * 30)

    # 计算FLOPs
    total_flops = calculate_autoencoder_flops(input_channels=input_channels, input_length=input_length)

    # 统计参数量
    param_stats = count_model_parameters(model)

    # 汇总信息
    print("\n" + "*" * 30)
    print("模型复杂度汇总")
    print("*" * 30)
    print(f"模型名称: AutoEncoder")
    print(f"输入形状: ({input_channels}, {input_length})")
    print(f"总FLOPs: {total_flops:,} ({total_flops / 1e6:.2f}M / {total_flops / 1e9:.4f}G)")
    print(f"总参数量: {param_stats['total_params']:,} ({param_stats['total_params'] / 1e6:.2f}M)")
    print(f"可训练参数: {param_stats['trainable_params']:,} ({param_stats['trainable_params'] / 1e6:.2f}M)")
    print("=" * 60 + "\n")

    return {
        'flops': total_flops,
        'params': param_stats
    }

# =========================================
# 模型评估
# =========================================
def evaluate(clf, X_test, y_test, model_type, measure_time=True):
    """
    扩展评估函数：新增KS、Purity（仅IF）、效应量指标
    Args:
        clf: 训练好的模型
        X_test: 测试集特征
        y_test: 测试集标签（1=正常，-1=异常）
        model_type: 模型类型（ocs/svdd/iforest/adaiforest）
        measure_time: 是否测量预测时间
    Returns:
        results: 包含所有评估指标的字典
    """
    # 1. 原有预测+时间测量逻辑
    if measure_time:
        start_time = time.time()

    y_pred = clf.predict(X_test)

    if measure_time:
        end_time = time.time()
        prediction_time = end_time - start_time
        print(f"预测时间: {prediction_time:.4f}秒 (样本数: {len(X_test)})")
        print(f"平均每样本预测时间: {prediction_time / len(X_test) * 1000:.4f}毫秒")

    # 2. 原有基础指标计算
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, pos_label=1)
    recall = recall_score(y_test, y_pred, pos_label=1)
    f1 = f1_score(y_test, y_pred, pos_label=1)
    false_positive_rate = sum((y_pred == 1) & (y_test == -1)) / sum(y_test == -1)

    # 3. 新增指标：提取异常分数，拆分正常/异常样本
    anomaly_scores = get_anomaly_scores(clf, X_test, model_type)
    normal_mask = (y_test == 1)
    abnormal_mask = (y_test == -1)
    normal_scores = anomaly_scores[normal_mask]
    abnormal_scores = anomaly_scores[abnormal_mask]

    # 4. 计算KS统计量
    ks_value = calculate_ks_statistic(normal_scores, abnormal_scores)

    # 5. 计算效应量（Cohen's d + Cliff's δ）
    cohens_d = calculate_cohens_d(normal_scores, abnormal_scores)
    cliffs_delta = calculate_cliffs_delta(normal_scores, abnormal_scores)

    # 6. 计算Purity（仅IForest/AdaptiveIForest）
    avg_purity = None
    max_purity = None
    if model_type in ["iforest", "adaiforest"]:
        avg_purity, max_purity = calculate_iforest_purity(clf, X_test, y_test)

    # 7. 打印所有指标
    print(f"准确率: {accuracy:.4f}")
    print(f"精准率: {precision:.4f}")
    print(f"召回率 (TPR): {recall:.4f}")
    print(f"F1 分数: {f1:.4f}")
    print(f"假阳性率 (FPR): {false_positive_rate:.4f}")
    print(f"KS统计量: {ks_value:.4f} (越大区分能力越强)")
    print(f"Cohen's d: {cohens_d:.4f} (>0.8为强区分度)")
    print(f"Cliff's δ: {cliffs_delta:.4f} (绝对值越大区分度越强)")
    if avg_purity is not None:
        print(f"IF叶节点平均异常纯度: {avg_purity:.4f}")
        print(f"IF叶节点最大异常纯度: {max_purity:.4f}")

    # 8. 整合所有结果
    results = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "false_positive_rate": false_positive_rate,
        "ks_statistic": ks_value,
        "cohens_d": cohens_d,
        "cliffs_delta": cliffs_delta
    }

    # 仅IF系列添加Purity
    if avg_purity is not None:
        results["if_avg_purity"] = avg_purity
        results["if_max_purity"] = max_purity

    # 原有时间指标
    if measure_time:
        results["prediction_time"] = prediction_time
        results["avg_prediction_time_per_sample"] = prediction_time / len(X_test)

    return results



# ===================== 新增：KS统计量 =====================
def calculate_ks_statistic(normal_scores, abnormal_scores):
    """
    计算KS统计量：max|F_normal(x) - F_abnormal(x)|
    Args:
        normal_scores: 正常样本的异常分数（一维数组）
        abnormal_scores: 异常样本的异常分数（一维数组）
    Returns:
        ks_value: KS统计量（0~1，越大区分能力越强）
    """
    # 合并所有分数并排序，作为CDF的x轴
    all_scores = np.concatenate([normal_scores, abnormal_scores])
    sorted_scores = np.sort(np.unique(all_scores))
    if len(sorted_scores) == 1:
        return 0.0  # 无区分度

    # 计算正常/异常样本的CDF
    cdf_normal = np.array([np.mean(normal_scores <= x) for x in sorted_scores])
    cdf_abnormal = np.array([np.mean(abnormal_scores <= x) for x in sorted_scores])

    # 计算最大绝对差值
    ks_value = np.max(np.abs(cdf_normal - cdf_abnormal))
    return ks_value


# ===================== 新增：IForest叶节点Purity =====================
def calculate_iforest_purity(if_model, X_test, y_test):
    """计算Isolation Forest叶节点异常纯度（仅支持sklearn IsolationForest）"""
    # 新增：判断是否为AdaptiveIsolationForest，若是则返回默认值
    if "AdaptiveIsolationForest" in str(type(if_model)) or "adaiforest" in str(type(if_model)).lower():
        print("提示：Adaptive Isolation Forest 无apply方法，跳过叶节点纯度计算")
        return 0.0, 0.0  # 返回默认值，不影响后续评估

    # 原逻辑（仅对原生IsolationForest生效）
    leaf_ids = if_model.apply(X_test)  # shape: (n_samples, n_estimators)
    avg_purity = 0.0
    max_purity = 0.0
    n_trees = leaf_ids.shape[1]

    # 遍历每棵树的叶节点计算纯度
    for tree_idx in range(n_trees):
        tree_leaf_ids = leaf_ids[:, tree_idx]
        unique_leaves = np.unique(tree_leaf_ids)
        tree_purities = []

        for leaf in unique_leaves:
            leaf_samples = np.where(tree_leaf_ids == leaf)[0]
            if len(leaf_samples) == 0:
                continue
            # 计算该叶节点中非法样本（异常）的比例（纯度）
            leaf_y = y_test[leaf_samples]
            anomaly_count = np.sum(leaf_y == -1)  # 假设-1是非法标签
            purity = anomaly_count / len(leaf_samples)
            tree_purities.append(purity)

        if tree_purities:
            avg_purity += np.mean(tree_purities)
            max_purity = max(max_purity, np.max(tree_purities))

    avg_purity /= n_trees
    return avg_purity, max_purity

# ===================== 新增：效应量（Cohen’s d / Cliff’s δ） =====================
def calculate_cohens_d(normal_data, abnormal_data):
    """
    计算Cohen's d：(均值差) / 合并标准差（适用于正态分布数据）
    Args:
        normal_data: 正常样本的特征/分数（一维数组）
        abnormal_data: 异常样本的特征/分数（一维数组）
    Returns:
        cohens_d: Cohen's d值（>0.8为强区分度）
    """
    n1, n2 = len(normal_data), len(abnormal_data)
    var1, var2 = np.var(normal_data, ddof=1), np.var(abnormal_data, ddof=1)
    pooled_std = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1 + n2 - 2))
    mean_diff = np.mean(normal_data) - np.mean(abnormal_data)
    cohens_d = mean_diff / pooled_std if pooled_std != 0 else 0.0
    return cohens_d

def calculate_cliffs_delta(normal_data, abnormal_data):
    """
    计算Cliff's δ：适用于非正态分布，衡量两组分布的重叠程度
    公式：δ = (P(X>Y) - P(X<Y))，范围[-1,1]，绝对值越大区分度越强
    Args:
        normal_data: 正常样本的特征/分数（一维数组）
        abnormal_data: 异常样本的特征/分数（一维数组）
    Returns:
        cliffs_delta: Cliff's δ值
    """
    # 生成所有两两比较对
    comparisons = []
    for x in normal_data:
        for y in abnormal_data:
            if x > y:
                comparisons.append(1)
            elif x < y:
                comparisons.append(-1)
            else:
                comparisons.append(0)
    cliffs_delta = np.mean(comparisons)
    return cliffs_delta

def get_anomaly_scores(clf, X_test, model_type):
    """
    提取模型的异常分数（统一规则：分数越高，越异常）
    Args:
        clf: 训练好的模型
        X_test: 测试集特征
        model_type: 模型类型（ocs/svdd/iforest/adaiforest）
    Returns:
        anomaly_scores: 异常分数数组（越高越异常）
    """
    if model_type in ["ocs", "svdd"]:
        # OCS/SVDD的decision_function越高→越正常 → 取负作为异常分数
        decision_vals = clf.decision_function(X_test)
        anomaly_scores = -decision_vals
    elif model_type in ["iforest", "adaiforest"]:
        # IForest的score_samples越低→越异常 → 取负作为异常分数
        score_samples = clf.score_samples(X_test)
        anomaly_scores = -score_samples
    else:
        raise ValueError(f"不支持的模型类型：{model_type}")
    return anomaly_scores

# =========================================
# 结果/模型保存与加载
# =========================================
def save_results(train_loss_history, test_loss_history, evaluation_results, args, extra_info=None):
    """保存 loss、评估指标以及额外信息到 CSV 文件"""
    os.makedirs("results/saved_csv", exist_ok=True)
    if args.channels == 2:
        filename = f"results_{args.range}_{args.data_split}_{args.normalization}_{args.channels}_{args.model}_lr{args.learning_rate}_epoch{args.epochs}_{args.two_channel}_idx{args.idx}_seed{args.seed}.csv"
    else:
        filename = f"results_{args.range}_{args.data_split}_{args.normalization}_{args.channels}_{args.model}_lr{args.learning_rate}_epoch{args.epochs}_{args.third_channel}_idx{args.idx}_seed{args.seed}.csv"

    filepath = os.path.join("results/saved_csv", filename)

    with open(filepath, 'w') as f:
        # 保存训练+测试损失（合并为一张表）
        if train_loss_history is not None and test_loss_history is not None \
                and len(train_loss_history) == len(test_loss_history):
            f.write("epoch,train_loss,test_loss\n")
            for i, (train_loss, test_loss) in enumerate(zip(train_loss_history, test_loss_history)):
                f.write(f"{i + 1},{train_loss},{test_loss}\n")
            f.write("\n")

        # 保存评估指标
        f.write("metric,value\n")
        for metric, value in evaluation_results.items():
            f.write(f"{metric},{value:.4f}\n")

        # 保存额外信息
        if extra_info is not None:
            if "svdd_iterations" in extra_info and isinstance(extra_info["svdd_iterations"], list):
                f.write("\nSVDD Iteration Logs:\n")
                f.write("iteration,log_info\n")
                for idx, line in enumerate(extra_info["svdd_iterations"]):
                    f.write(f"{idx + 1},{line.strip()}\n")
            else:
                f.write("\nextra_metric,value\n")
                for metric, value in extra_info.items():
                    f.write(f"{metric},{value}\n")

    print(f"训练损失、测试损失、评估结果及额外信息已保存到 {filepath}")

def save_classifier(clf, args):
    """保存分类器模型（OCS/SVDD/IForest/AdaIForest）"""
    os.makedirs("model_pkl", exist_ok=True)

    if args.channels == 2:
        clf_filename = f"{args.range}_{args.data_split}_{args.normalization}_{args.channels}_{args.model}_{args.two_channel}_idx{args.idx}_seed{args.seed}.pkl"
    else:
        clf_filename = f"{args.range}_{args.data_split}_{args.normalization}_{args.channels}_{args.model}_{args.third_channel}_idx{args.idx}_seed{args.seed}.pkl"

    clf_path = os.path.join("model_pkl", clf_filename)
    joblib.dump(clf, clf_path)
    print(f"{args.model.upper()} 模型已保存到 {clf_path}")
    return clf_path

def load_classifier(args):
    """加载分类器模型（OCS/SVDD/IForest/AdaIForest）"""
    if args.channels == 2:
        clf_filename = f"{args.range}_{args.data_split}_{args.normalization}_{args.channels}_{args.model}_{args.two_channel}_idx{args.idx}_seed{args.seed}.pkl"
    else:
        clf_filename = f"{args.range}_{args.data_split}_{args.normalization}_{args.channels}_{args.model}_{args.third_channel}_idx{args.idx}_seed{args.seed}.pkl"

    clf_path = os.path.join("model_pkl", clf_filename)

    if not os.path.exists(clf_path):
        raise FileNotFoundError(f"找不到模型文件: {clf_path}\n请先运行训练模式 (--mode train)")

    clf = joblib.load(clf_path)
    print(f"已加载 {args.model.upper()} 模型：{clf_path}")
    return clf

def load_pretrained_model(model_path, input_channels):
    """加载预训练的AutoEncoder模型"""
    from model import AutoEncoder  # 避免循环导入
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = AutoEncoder(input_channels=input_channels)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    print(f"已加载预训练模型：{model_path}")
    return model