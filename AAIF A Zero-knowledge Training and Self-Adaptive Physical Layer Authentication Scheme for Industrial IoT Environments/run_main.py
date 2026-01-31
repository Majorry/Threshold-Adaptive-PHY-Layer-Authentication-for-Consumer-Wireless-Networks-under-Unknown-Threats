import numpy as np

from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import cvxopt
import cvxopt.solvers
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import random
import io
from contextlib import redirect_stdout
import time

import os


# -----------------------------
# 数据加载函数：从 .npy 文件加载数据
# -----------------------------
def load_csi_data(data_dir, legal_file, illegal_file):
    legal_path = os.path.join(data_dir, legal_file)
    illegal_path = os.path.join(data_dir, illegal_file)
    legal_data = np.load(legal_path)  # shape: (samples, 2, 8188)
    illegal_data = np.load(illegal_path)  # shape: (samples, 2, 8188)
    return legal_data, illegal_data


def preprocess_data(data):
    return data.reshape(data.shape[0], 2, 8188)


# =========================================
# 1. 基础函数：初始化模型参数
# =========================================
def init_weights(module):
    """对不同层类型使用合适的初始化策略"""
    if isinstance(module, nn.Linear):
        # Kaiming 初始化（适用于LeakyReLU）
        nn.init.kaiming_normal_(module.weight, a=0.2, nonlinearity='leaky_relu')
        if module.bias is not None:
            nn.init.constant_(module.bias, 0.0)

def count_conv1d_flops(in_channels, out_channels, kernel_size, input_length, stride=1, padding=0):
    """
    计算Conv1d层的FLOPs
    """
    # 计算输出长度
    output_length = (input_length + 2 * padding - kernel_size) // stride + 1
    # FLOPs = 输出特征图大小 × (卷积核大小 × 输入通道数 × 2)
    # 2 表示乘法和加法各一次
    flops = output_length * out_channels * (kernel_size * in_channels * 2)
    return flops, output_length
def count_convtranspose1d_flops(in_channels, out_channels, kernel_size, input_length, stride=1, padding=0, output_padding=0):
    """
    计算ConvTranspose1d层的FLOPs
    """
    # 计算输出长度
    output_length = (input_length - 1) * stride - 2 * padding + kernel_size + output_padding
    # FLOPs计算方式与Conv1d类似
    flops = output_length * out_channels * (kernel_size * in_channels * 2)
    return flops, output_length

def count_batchnorm1d_flops(num_features, input_length):
    """
    计算BatchNorm1d层的FLOPs
    """
    # BatchNorm: 每个元素需要4次操作 (减均值、除标准差、乘gamma、加beta)
    flops = input_length * num_features * 4
    return flops

def count_activation_flops(input_length, num_features):
    """
    计算激活函数的FLOPs (LeakyReLU, Tanh等)
    """
    # 激活函数通常计为1次操作per元素
    flops = input_length * num_features
    return flops


def calculate_autoencoder_flops(input_channels=2, input_length=8188):
    """
    计算AutoEncoder的总FLOPs
    """
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
    """
    统计模型的参数量
    """
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
    """
    打印模型的FLOPs和参数统计信息

    Args:
        model: AutoEncoder模型实例
        input_channels: 输入通道数
        input_length: 输入序列长度
    """
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
# 2. 自编码器模型（AE）
# =========================================
class AutoEncoder(nn.Module):
    def __init__(self, input_channels=2):
        super(AutoEncoder, self).__init__()
        # Encoder - 输入形状: (batch, 2, 8188)
        self.encoder = nn.Sequential(
            # (2, 8188) -> (16, 4094)
            nn.Conv1d(input_channels, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(True),

            # (16, 4094) -> (32, 2047)
            nn.Conv1d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(True),

            # (32, 2047) -> (64, 1024)
            nn.Conv1d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(True),

            # (64, 1024) -> (128, 512)
            nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(True),

        )

        # Decoder（对称结构）- 输出形状: (batch, 2, 8188)
        self.decoder = nn.Sequential(

            # (128, 512) -> (64, 1024)
            nn.ConvTranspose1d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(True),

            # (64, 1024) -> (32, 2047)
            nn.ConvTranspose1d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=0),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(True),

            # (32, 2047) -> (16, 4094)
            nn.ConvTranspose1d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(True),

            # (16, 4094) -> (2, 8188)
            nn.ConvTranspose1d(16, input_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Tanh()
        )

        # 初始化参数
        self._initialize_weights()

    def _initialize_weights(self):
        self.apply(init_weights)

    def forward(self, x):
        latent = self.encoder(x)
        recon = self.decoder(latent)
        return recon

    def encode(self, x):
        return self.encoder(x)


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
# 评估模型性能 (增加时间测量)
# -----------------------------
def evaluate(clf, X_test, y_test, measure_time=True):
    if measure_time:
        start_time = time.time()

    y_pred = clf.predict(X_test)

    if measure_time:
        end_time = time.time()
        prediction_time = end_time - start_time
        print(f"预测时间: {prediction_time:.4f}秒 (样本数: {len(X_test)})")
        print(f"平均每样本预测时间: {prediction_time / len(X_test) * 1000:.4f}毫秒")

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, pos_label=1)
    recall = recall_score(y_test, y_pred, pos_label=1)
    f1 = f1_score(y_test, y_pred, pos_label=1)
    false_positive_rate = sum((y_pred == 1) & (y_test == -1)) / sum(y_test == -1)

    print(f"准确率: {accuracy:.4f}")
    print(f"精准率: {precision:.4f}")
    print(f"召回率 (TPR): {recall:.4f}")
    print(f"F1 分数: {f1:.4f}")
    print(f"假阳性率 (FPR): {false_positive_rate:.4f}")

    results = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "false_positive_rate": false_positive_rate
    }

    if measure_time:
        results["prediction_time"] = prediction_time
        results["avg_prediction_time_per_sample"] = prediction_time / len(X_test)

    return results


# -----------------------------
# 保存 loss、评估指标以及额外信息到 CSV 文件
# -----------------------------
def save_results(train_loss_history, test_loss_history, evaluation_results, args, extra_info=None):
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


#
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


# ✅ 新增：加载分类器模型
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

# -----------------------------
# 加载预训练模型
# -----------------------------
def load_pretrained_model(model_path, input_channels):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = AutoEncoder(input_channels=input_channels)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    print(f"已加载预训练模型：{model_path}")
    return model


# -----------------------------
# SVDD模型实现
# -----------------------------
class SVDD:
    def __init__(self, C, kernel="rbf", gamma="scale"):
        self.C = C
        self.kernel = kernel
        self.gamma = gamma
        self.iteration_logs = []  # 用于保存每次迭代的日志

    def fit(self, X):
        # ============ 新增：处理多维数据 ============
        original_shape = X.shape
        print(f"SVDD fit - 原始输入形状: {original_shape}")

        # 如果是多维数据，展平为 (样本量, 特征维度)
        if len(X.shape) > 2:
            n_samples = X.shape[0]
            X = X.reshape(n_samples, -1)
            print(f"SVDD fit - 展平后形状: {X.shape}")
        # ==========================================

        self.X_train = X

        # 计算 RBF 核矩阵
        if self.kernel == "rbf":
            if self.gamma == "scale":
                self.gamma_val = 1 / (X.shape[1] * X.var())
            else:
                self.gamma_val = self.gamma
            self.K = np.exp(-self.gamma_val * np.linalg.norm(X[:, None] - X, axis=2) ** 2).astype(np.double)
        else:
            raise ValueError("暂不支持的核函数")

        print(f"核矩阵 K 的形状: {self.K.shape}")

        n = X.shape[0]
        P = cvxopt.matrix(2 * self.K.astype(np.double))
        q = cvxopt.matrix(-np.diag(self.K).astype(np.double))
        A = cvxopt.matrix(np.ones((1, n), dtype=np.double))
        b = cvxopt.matrix(1.0, tc='d')
        G = cvxopt.matrix(np.vstack((-np.eye(n), np.eye(n))).astype(np.double))
        h = cvxopt.matrix(np.hstack((np.zeros(n), np.ones(n) * self.C)).astype(np.double))

        # 捕获 cvxopt 求解器的输出
        buf = io.StringIO()
        with redirect_stdout(buf):
            sol = cvxopt.solvers.qp(P, q, G, h, A, b)
        output = buf.getvalue()
        self.iteration_logs = output.splitlines()

        alphas = np.array(sol['x']).flatten()
        self.alphas = alphas

        support_vector_indices = np.where((alphas > 1e-5) & (alphas < self.C - 1e-5))[0]
        if len(support_vector_indices) == 0:
            support_vector_indices = np.where(alphas > 1e-5)[0]

        sv_index = support_vector_indices[0]
        self.R2 = self.K[sv_index, sv_index] - 2 * np.sum(alphas * self.K[:, sv_index]) + \
                  np.sum(np.outer(alphas, alphas) * self.K)

    def decision_function(self, X_test):
        # ============ 新增：处理多维数据 ============
        original_shape = X_test.shape
        print(f"SVDD decision_function - 原始输入形状: {original_shape}")

        # 如果是多维数据，展平为 (样本量, 特征维度)
        if len(X_test.shape) > 2:
            n_samples = X_test.shape[0]
            X_test = X_test.reshape(n_samples, -1)
            print(f"SVDD decision_function - 展平后形状: {X_test.shape}")
        # ==========================================

        if X_test.shape[1] != self.X_train.shape[1]:
            raise ValueError(f"X_test 和 self.X_train 的特征维度不匹配: {X_test.shape[1]} vs {self.X_train.shape[1]}")

        if self.kernel == "rbf":
            K_test = np.exp(-self.gamma_val * np.linalg.norm(X_test[:, None] - self.X_train, axis=2) ** 2)
        else:
            raise ValueError("暂不支持的核函数")

        distances = 1 - 2 * np.dot(K_test, self.alphas) + np.sum(np.outer(self.alphas, self.alphas) * self.K)
        return self.R2 - distances

    def predict(self, X_test):
        decision_values = self.decision_function(X_test)
        y_pred = np.where(decision_values >= 0, 1, -1)
        return y_pred


class AdaptiveIsolationForest(IsolationForest):
    """
    自适应 Isolation Forest：
    阈值根据训练数据分数自动计算，不需要传入 contamination 参数。
    基于 IQR 方法：
        阈值 = Q3 + 1.5 * IQR
    """
    def __init__(self, **kwargs):
        super().__init__(contamination='auto', **kwargs)
        self.threshold_ = None  # 自动阈值

    def fit(self, X, y=None):
        super().fit(X, y)
        # 得分越大越正常，越小越异常
        scores = -self.score_samples(X)  # 注意取负号，使得越大越异常
        Q1 = np.percentile(scores, 25)
        Q3 = np.percentile(scores, 75)
        IQR = Q3 - Q1
        self.threshold_ = Q3 + 1.5 * IQR  # 超过此阈值判异常
        print(f"[Adaptive IF] 自动计算阈值: {self.threshold_:.6f} (基于 IQR)")
        return self

    def predict(self, X):
        scores = -self.score_samples(X)
        return np.where(scores > self.threshold_, -1, 1)  # 1=正常, -1=异常

    def decision_function(self, X):
        scores = -self.score_samples(X)
        return self.threshold_ - scores

# -----------------------------
# 主函数：训练/测试流程
# -----------------------------
def run_model(args):
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    if args.range in ['400_oats_82', '400_oats_55', '400_oats_32', '40_oats_55']:
        scenario = 'OATS'
    elif args.range in ['300_aap1_82', '300_aap1_55', '300_aap1_32', '50_aap1_55', '200_aap1_55', '100_aap1_55']:
        scenario = 'AAP1'
    elif args.range in ['300_aap2_55', '300_aap2_32', '300_aap2_82', '50_aap2_55', '200_aap2_55', '100_aap2_55']:
        scenario = 'AAP2'
    elif args.range in ['300_gburg_55', '300_gburg_82', '300_gburg_32', ]:
        scenario = 'GBurg'

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

    X_train_legal, _ = load_csi_data(train_data_dir, train_legal_file, train_illegal_file)
    X_train_legal = preprocess_data(X_train_legal)
    X_test_legal, X_test_illegal = load_csi_data(test_data_dir, test_legal_file, test_illegal_file)
    X_test_legal = preprocess_data(X_test_legal)
    X_test_illegal = preprocess_data(X_test_illegal)

    # 打印原始数据大小信息
    print(f"训练集合法样本形状: {X_train_legal.shape}")
    print(f"测试集合法样本形状: {X_test_legal.shape}")
    print(f"测试集非法样本形状: {X_test_illegal.shape}")

    y_test = np.concatenate([np.ones(len(X_test_legal)), -np.ones(len(X_test_illegal))])
    X_test_combined = np.concatenate([X_test_legal, X_test_illegal])
    train_loss_history = None
    test_loss_history = None

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

    # ✅ 保存分类器模型
    save_classifier(clf, args)

    # =============================================================

    # 模型评估
    print(f"\n开始评估 {args.model.upper()} 模型...")
    if args.model != "adaiforest":
        if len(X_test_model.shape) > 2:
            print(f"展平前数据形状: {X_test_model.shape}")
            X_test_model = X_test_model.reshape(X_test_model.shape[0], -1)
            print(f"展平后数据形状: {X_test_model.shape}")
    else:
        print("No flatten!")

    # ✅ 加载预训练的分类器
    clf = load_classifier(args)
    evaluation_results = evaluate(clf, X_test_model, y_test)

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
    import argparse

    parser = argparse.ArgumentParser(
        description="基于CSI的合法/非法识别系统，支持AE特征提取，并结合OC-SVM、SVDD或Isolation Forest进行认证。"
    )
    parser.add_argument('--range', type=str,
                        choices=['300_aap1_55', '300_aap1_32', '300_aap1_82', '200_aap1_55', '100_aap1_55', '50_aap1_55',
                                 '300_aap2_55','300_aap2_82', '50_aap2_55', '300_aap2_32', '200_aap2_55', '100_aap2_55',
                                 '300_gburg_55', '300_gburg_82','300_gburg_32',
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

    # 新增：Isolation Forest 参数
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