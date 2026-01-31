# -*- coding: gbk -*-

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import random
import time
import argparse
import os
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
import joblib

# 导入自定义模块
from data_loader import load_csi_data, preprocess_data
from model import AutoEncoder, SVDD, AdaptiveIsolationForest, init_weights
from utils import (
    print_model_complexity, evaluate, save_results,
    save_classifier, load_classifier, load_pretrained_model, plot_tsne
)



# -----------------------------
# 自编码器训练函数
# -----------------------------
def train_autoencoder(X_train, X_test, epochs, batch_size, learning_rate):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 训练集数据准备 - 数据已经是 (N, 2, 8188) 形状
    X_tensor = torch.FloatTensor(X_train).to(device)
    train_dataset = TensorDataset(X_tensor, X_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # 测试集数据准备
    X_test_tensor = torch.FloatTensor(X_test).to(device)
    test_dataset = TensorDataset(X_test_tensor, X_test_tensor)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 输入通道数为2
    input_channels = X_train.shape[1]

    model = AutoEncoder(input_channels=input_channels).to(device)
    # 打印模型复杂度信息
    model_complexity = print_model_complexity(model, input_channels=input_channels, input_length=8188)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_loss_history = []
    test_loss_history = []

    for epoch in range(epochs):
        # 训练阶段
        model.train()
        total_train_loss = 0.0
        for data, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
        avg_train_loss = total_train_loss / len(train_loader)
        train_loss_history.append(avg_train_loss)

        # 测试阶段（验证损耗）
        model.eval()
        total_test_loss = 0.0
        with torch.no_grad():
            for data, targets in test_loader:
                outputs = model(data)
                loss = criterion(outputs, targets)
                total_test_loss += loss.item()
        avg_test_loss = total_test_loss / len(test_loader)
        test_loss_history.append(avg_test_loss)

        print(f"AE Epoch [{epoch + 1}/{epochs}], "
              f"Train Loss: {avg_train_loss:.6f}, "
              f"Test Loss: {avg_test_loss:.6f}")

    return model, train_loss_history, test_loss_history


# -----------------------------
# 特征提取函数
# -----------------------------
def get_ae_features(model, data):
    """
    提取编码器特征
    输入: data shape = (N, 2, 8188)
    输出: features shape = (N, 128, 512) -> 展平为 (N, 128*512)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    with torch.no_grad():
        # 数据已经是 (N, 2, 8188) 形状,直接转换为tensor
        data_tensor = torch.FloatTensor(data).to(device)
        features = model.encode(data_tensor)
        # 展平特征: (N, C, H, W) -> (N, C*H*W)
        features_flat = features.view(features.size(0), -1)
    return features_flat.cpu().numpy()


# -----------------------------
# 主函数：训练/测试流程
# -----------------------------
def run_model(args):
    # 设置随机种子
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # 场景判断
    if args.range in ['400_oats_82', '400_oats_55', '400_oats_32', '40_oats_55']:
        scenario = 'OATS'
    elif args.range in ['300_aap1_82', '300_aap1_55', '300_aap1_32', '50_aap1_55', '200_aap1_55', '100_aap1_55']:
        scenario = 'AAP1'
    elif args.range in ['300_aap2_55', '300_aap2_32', '300_aap2_82', '50_aap2_55', '200_aap2_55', '100_aap2_55']:
        scenario = 'AAP2'
    elif args.range in ['300_gburg_55', '300_gburg_82', '300_gburg_32']:
        scenario = 'GBurg'
    else:
        raise ValueError(f"未知的range: {args.range}")

    # 数据目录及文件配置
    train_data_dir = f"preprocess/processed_data/{scenario}/{args.count}/train"
    test_data_dir = f"preprocess/processed_data/{scenario}/{args.count}/test"

    if args.channels == 2:
        train_legal_file = f"legal_{args.range}_{args.data_split}_{args.normalization}_channel{args.channels}_{args.two_channel}_{args.idx}.npy"
        train_illegal_file = f"illegal_{args.range}_{args.data_split}_{args.normalization}_channel{args.channels}_{args.two_channel}_{args.idx}.npy"
        test_legal_file = f"legal_{args.range}_{args.data_split}_{args.normalization}_channel{args.channels}_{args.two_channel}_{args.idx}.npy"
        test_illegal_file = f"illegal_{args.range}_{args.data_split}_{args.normalization}_channel{args.channels}_{args.two_channel}_{args.idx}.npy"
    else:
        train_legal_file = f"legal_{args.range}_{args.data_split}_{args.normalization}_channel{args.channels}_{args.third_channel}_{args.idx}.npy"
        train_illegal_file = f"illegal_{args.range}_{args.data_split}_{args.normalization}_channel{args.channels}_{args.third_channel}_{args.idx}.npy"
        test_legal_file = f"legal_{args.range}_{args.data_split}_{args.normalization}_channel{args.channels}_{args.third_channel}_{args.idx}.npy"
        test_illegal_file = f"illegal_{args.range}_{args.data_split}_{args.normalization}_channel{args.channels}_{args.third_channel}_{args.idx}.npy"

    # 加载数据
    X_train_legal, _ = load_csi_data(train_data_dir, train_legal_file, train_illegal_file)
    X_train_legal = preprocess_data(X_train_legal)
    X_test_legal, X_test_illegal = load_csi_data(test_data_dir, test_legal_file, test_illegal_file)
    X_test_legal = preprocess_data(X_test_legal)
    X_test_illegal = preprocess_data(X_test_illegal)

    # 打印原始数据大小信息
    print(f"训练集合法样本形状: {X_train_legal.shape}")
    print(f"测试集合法样本形状: {X_test_legal.shape}")
    print(f"测试集非法样本形状: {X_test_illegal.shape}")


    # 构建测试集标签
    y_test = np.concatenate([np.ones(len(X_test_legal)), -np.ones(len(X_test_illegal))])
    X_test_combined = np.concatenate([X_test_legal, X_test_illegal])
    train_loss_history = None
    test_loss_history = None

    # ==========================================
    # [新增] 1. 训练前：原始数据的 t-SNE 可视化
    # ==========================================
    # 我们使用测试集(包含Normal和Abnormal)来观察原始数据的分布情况
    # 这里的 X_test_combined 形状为 (N, 2, 8188)，plot_tsne 会自动展平处理
    tsne_save_path_raw = f"results/figures/{args.range}_{args.model}_tsne_raw.png"

    plot_tsne(X_test_combined, y_test,
              save_path=tsne_save_path_raw,
              perplexity=5,  # 低 perplexity -> 视觉上更分散/细碎
              figsize=(5, 4),  # 小画布
              point_size=30,  # 大点
              legend_fontsize=10)

    print("\n========== 计算AE输入前（原始预处理数据）的区分度指标 ==========")
    # 1. 拆分测试集合法/非法样本（AE输入前的原始预处理数据）
    # X_test_legal/X_test_illegal 是AE输入前的最终数据，已完成预处理
    X_test_normal = X_test_legal  # 合法样本（AE输入前）
    X_test_abnormal = X_test_illegal  # 非法样本（AE输入前）

    # 2. 展平多维特征（适配一维数值提取，如CSI数据：(n, channels, length) → (n, channels*length)）
    normal_flat = X_test_normal.reshape(X_test_normal.shape[0], -1)
    abnormal_flat = X_test_abnormal.reshape(X_test_abnormal.shape[0], -1)

    # 3. 提取对比数值（样本特征均值，简单通用）
    normal_scores = np.mean(normal_flat, axis=1)  # 每个合法样本的特征均值
    abnormal_scores = np.mean(abnormal_flat, axis=1)  # 每个非法样本的特征均值

    # 4. 计算KS/Cohen's d/Cliff's δ（复用utils中的计算逻辑）
    # 注意：需确保utils.py中已实现这三个函数，若未封装可直接复制计算逻辑
    from scipy.stats import ks_2samp, mannwhitneyu

    # KS统计量
    ks_stat, _ = ks_2samp(normal_scores, abnormal_scores)

    # Cohen's d
    n1, n2 = len(normal_scores), len(abnormal_scores)
    mean1, mean2 = np.mean(normal_scores), np.mean(abnormal_scores)
    var1, var2 = np.var(normal_scores, ddof=1), np.var(abnormal_scores, ddof=1)
    sp = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1 + n2 - 2)) if (n1 + n2 - 2) != 0 else 0
    cohens_d = (mean1 - mean2) / sp if sp != 0 else 0

    # Cliff's δ
    if n1 * n2 == 0:
        cliffs_delta = 0
    else:
        u_stat, _ = mannwhitneyu(normal_scores, abnormal_scores, alternative='two-sided')
        cliffs_delta = (2 * u_stat / (n1 * n2)) - 1

    # 5. 打印AE输入前的指标结果
    print(f"AE输入前 - KS统计量: {ks_stat:.4f} (越大区分能力越强)")
    print(f"AE输入前 - Cohen's d: {cohens_d:.4f} (>0.8为强区分度)")
    print(f"AE输入前 - Cliff's δ: {cliffs_delta:.4f} (绝对值越大区分度越强)")


    # 自编码器特征提取部分
    if args.use_autoencoder:
        if args.mode == "train":
            print("训练 AE 进行去噪与特征提取...")
            model, train_loss_history, test_loss_history = train_autoencoder(
                X_train_legal, X_test_legal,
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate)
            X_train_model = get_ae_features(model, X_train_legal)
            X_test_legal_model = get_ae_features(model, X_test_legal)
            X_test_illegal_model = get_ae_features(model, X_test_illegal)

            # 保存模型
            os.makedirs("model_pt", exist_ok=True)
            model_save_filename = f"{args.range}_{args.data_split}_{args.normalization}_{args.channels}_ae_{args.model}.pt"
            model_save_path = os.path.join("model_pt", model_save_filename)
            torch.save(model.state_dict(), model_save_path)
            print(f"训练结束，模型已保存到 {model_save_path}")

        elif args.mode == "test":
            model_save_filename = f"50_aap1_55_{args.data_split}_{args.normalization}_{args.channels}_ae_{args.model}.pt"
            model_save_path = os.path.join("model_pt", model_save_filename)
            model = load_pretrained_model(model_save_path, X_train_legal.shape[1])
            X_train_model = get_ae_features(model, X_train_legal)
            X_test_legal_model = get_ae_features(model, X_test_legal)
            X_test_illegal_model = get_ae_features(model, X_test_illegal)
        else:
            raise ValueError("未知的模式，请选择 'train' 或 'test'")

        X_test_model = np.concatenate([X_test_legal_model, X_test_illegal_model])
        # ==========================================
        # [新增] 2. 训练后：潜在空间特征的 t-SNE 可视化
        # ==========================================
        # 这里的 X_test_model 已经是 get_ae_features 展平后的 (N, 128*512)
        tsne_save_path_latent = f"results/figures/{args.range}_{args.model}_tsne_latent.png"

        plot_tsne(X_test_model, y_test,
                  save_path=tsne_save_path_latent,
                  perplexity=50,  # 高 perplexity -> 视觉上更聚集/成团
                  figsize=(5, 4),  # 小画布
                  point_size=30,  # 大点
                  legend_fontsize=10)

    else:
        print("不使用自编码器，直接使用原始数据...")
        X_train_model = X_train_legal
        X_test_model = X_test_combined

    # ============ 模型训练部分 ============
    extra_info = {}

    if args.model == "ocs":
        print("使用 One-Class SVM 模型")
        if len(X_train_model.shape) > 2:
            print(f"展平前数据形状: {X_train_model.shape}")
            X_train_model = X_train_model.reshape(X_train_model.shape[0], -1)
            print(f"展平后数据形状: {X_train_model.shape}")

        clf = OneClassSVM(nu=args.svm_nu, kernel="rbf", gamma="scale")
        clf.fit(X_train_model)

    elif args.model == "svdd":
        print("使用 SVDD 模型")
        clf = SVDD(C=args.svdd_C, kernel="rbf", gamma="scale")
        clf.fit(X_train_model)
        extra_info["svdd_iterations"] = clf.iteration_logs

    elif args.model == "iforest":
        print("使用 Isolation Forest 模型")
        print(f"参数设置: n_estimators={args.if_n_estimators}, "
              f"max_samples={args.if_max_samples}, "
              f"contamination={args.if_contamination}, "
              f"random_state={args.seed}")
        if len(X_train_model.shape) > 2:
            print(f"展平前数据形状: {X_train_model.shape}")
            X_train_model = X_train_model.reshape(X_train_model.shape[0], -1)
            print(f"展平后数据形状: {X_train_model.shape}")
        # 标准 Isolation Forest 实例化
        clf = IsolationForest(
            n_estimators=args.if_n_estimators,
            max_samples=args.if_max_samples,
            contamination=args.if_contamination,
            random_state=args.seed,
            n_jobs=-1
        )
        clf.fit(X_train_model)

        # 保存 Isolation Forest 的统计信息
        extra_info["iforest_threshold"] = clf.threshold_ if hasattr(clf, "threshold_") else None
        extra_info["iforest_score_mean"] = float(np.mean(clf.score_samples(X_train_model)))
        extra_info["iforest_score_std"] = float(np.std(clf.score_samples(X_train_model)))

    elif args.model == "adaiforest":
        print("使用 Adaptive Isolation Forest 模型")
        print(f"参数设置: n_estimators={args.if_n_estimators}, "
              f"max_samples={args.if_max_samples}, "
              f"random_state={args.seed}")

        clf = AdaptiveIsolationForest(
            n_estimators=args.if_n_estimators,
            max_samples=args.if_max_samples,
            random_state=args.seed,
            n_jobs=-1
        )
        clf.fit(X_train_model)

        # 保存 Isolation Forest 的一些统计信息
        extra_info["adaptive_threshold"] = clf.threshold_

    else:
        raise ValueError("未识别的模型选择，请选择 'ocs'、'svdd' 或 'adaiforest'。")

    # 保存分类器模型
    save_classifier(clf, args)

    # 模型评估
    print(f"\n开始评估 {args.model.upper()} 模型...")
    if args.model != "adaiforest":
        if len(X_test_model.shape) > 2:
            print(f"展平前数据形状: {X_test_model.shape}")
            X_test_model = X_test_model.reshape(X_test_model.shape[0], -1)
            print(f"展平后数据形状: {X_test_model.shape}")
    else:
        print("No flatten!")

    # 加载预训练的分类器
    clf = load_classifier(args)
    # 关键修改：传入model_type（args.model）
    evaluation_results = evaluate(clf, X_test_model, y_test, model_type=args.model)

    # 保存结果
    if args.save_csv and args.mode == "train":
        if not args.use_autoencoder:
            train_loss_history = []
            test_loss_history = []
        save_results(train_loss_history, test_loss_history, evaluation_results, args, extra_info=extra_info)


# -----------------------------
# 命令行参数解析
# -----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="基于CSI的合法/非法识别系统，支持AE特征提取，并结合OC-SVM、SVDD或Isolation Forest进行认证。"
    )
    parser.add_argument('--range', type=str,
                        choices=['300_aap1_55', '300_aap1_32', '300_aap1_82', '200_aap1_55', '100_aap1_55',
                                 '50_aap1_55',
                                 '300_aap2_55', '300_aap2_82', '50_aap2_55', '300_aap2_32', '200_aap2_55',
                                 '100_aap2_55',
                                 '300_gburg_55', '300_gburg_82', '300_gburg_32',
                                 '400_oats_55', '400_oats_82', '400_oats_32', '40_oats_55'],
                        default='400_oats', help='数据范围与场景选择')
    parser.add_argument('--data_split', default='random_82', type=str,
                        choices=['random_82', 'random_55', 'random_28', 'sequence', '2_1', '3_1', '4_1', '5_1',
                                 '6_1'],
                        help='数据划分方式')
    parser.add_argument('--idx', type=int, default=1, help='数据索引')
    parser.add_argument('--channels', type=int, choices=[2, 3], default=2, help='通道数：2 或 3')
    parser.add_argument('--two_channel', type=str, default='real_imag',
                        choices=['real_imag', 'magnitude_phase'], help='二通道组成选项')
    parser.add_argument('--third_channel', type=str, default='all_magnitude',
                        choices=['magnitude', 'phase', 'all_magnitude', 'all_phase'], help='第三通道选项')

    parser.add_argument('--normalization', type=str, choices=['none', 'minmax', 'standardize'], default='minmax',
                        help="数据归一化方法")

    parser.add_argument('--model', type=str, choices=['ocs', 'svdd', 'iforest', 'adaiforest'], default='ocs',
                        help='认证模型选择："ocs"表示One-Class SVM，"svdd"表示SVDD，"iforest"表示Isolation Forest')

    parser.add_argument('--use_autoencoder', action='store_true',
                        help='是否使用自编码器进行去噪并提取特征')
    parser.add_argument('--save_csv', action='store_true', help='是否保存 loss 历史记录和评估结果为 CSV 文件')
    parser.add_argument('--epochs', type=int, default=100, help='训练轮数')
    parser.add_argument('--count', type=int, help='用户对数量')
    parser.add_argument('--batch_size', type=int, default=32, help='训练批次大小')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='学习率')

    # OC-SVM 参数
    parser.add_argument('--svm_nu', type=float, default=0.9, help='One-Class SVM的nu参数')

    # SVDD 参数
    parser.add_argument('--svdd_C', type=float, default=0.1, help='SVDD中的C参数')

    # Isolation Forest 参数
    parser.add_argument('--if_contamination', type=float, default=0.5)
    parser.add_argument('--if_n_estimators', type=int, default=100,
                        help='Isolation Forest的树的数量')
    parser.add_argument('--if_max_samples', type=str, default='auto',
                        help='Isolation Forest每棵树的最大样本数，可以是整数或"auto"')

    parser.add_argument('--pca_components', type=int, default=2, help='使用PCA时降维的目标维度（通常为2）')
    parser.add_argument('--seed', type=int, default=42, help='随机种子，保证实验结果可复现')
    parser.add_argument('--mode', type=str, choices=['train', 'test'], default='train',
                        help='模式选择：train 训练 / test 测试')

    args = parser.parse_args()

    # 处理 if_max_samples 参数
    if args.if_max_samples != 'auto':
        try:
            args.if_max_samples = int(args.if_max_samples)
        except ValueError:
            print(f"警告: if_max_samples 参数 '{args.if_max_samples}' 无效，使用默认值 'auto'")
            args.if_max_samples = 'auto'

    run_model(args)