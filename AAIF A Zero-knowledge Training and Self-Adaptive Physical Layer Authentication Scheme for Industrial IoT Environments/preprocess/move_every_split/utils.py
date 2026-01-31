import os
import numpy as np
import torch
from parser import parse_args
#---------------------------Normalization the data---------------------------

args = parse_args()
def normalize_with_minmax_per_channel(data, train_min, train_max):
    # 确保 data 是 PyTorch 张量
    if isinstance(data, np.ndarray):
        data = torch.from_numpy(data).float()  # 转换为 PyTorch 张量

    # 确保 train_min 和 train_max 是 PyTorch 张量
    if not isinstance(train_min, torch.Tensor):
        train_min = torch.tensor(train_min, dtype=data.dtype, device=data.device).view(1, -1, 1)
    else:
        train_min = train_min.to(dtype=data.dtype, device=data.device).view(1, -1, 1)

    if not isinstance(train_max, torch.Tensor):
        train_max = torch.tensor(train_max, dtype=data.dtype, device=data.device).view(1, -1, 1)
    else:
        train_max = train_max.to(dtype=data.dtype, device=data.device).view(1, -1, 1)

    # 归一化操作
    normalized_data = (data - train_min) / (train_max - train_min + 1e-8)
    return normalized_data

def standardize_per_channel(data, train_mean, train_std):
    # 将 data 转换为 PyTorch 张量（如果它还不是）
    if not isinstance(data, torch.Tensor):
        data = torch.from_numpy(data)
    # 将 train_mean 转换为 PyTorch 张量
    if not isinstance(train_mean, torch.Tensor):
        # 如果 train_mean 是 NumPy 数组，转换为 PyTorch 张量
        if isinstance(train_mean, np.ndarray):
            train_mean = torch.from_numpy(train_mean)
        else:
            train_mean = torch.tensor(train_mean)
        # 确保数据类型与 data 一致
        train_mean = train_mean.to(dtype=data.dtype).view(1, -1, 1)
    # 将 train_std 转换为 PyTorch 张量
    if not isinstance(train_std, torch.Tensor):
        # 如果 train_std 是 NumPy 数组，转换为 PyTorch 张量
        if isinstance(train_std, np.ndarray):
            train_std = torch.from_numpy(train_std)
        else:
            train_std = torch.tensor(train_std)
        # 确保数据类型与 data 一致
        train_std = train_std.to(dtype=data.dtype).view(1, -1, 1)
    # 标准化数据
    standardized_data = (data - train_mean) / (train_std + 1e-8)
    return standardized_data

#---------------------------Compute magnitude---------------------------
def compute_magnitude(data):
    if data.shape[1] != 2:
        raise ValueError("输入数据的第二维 channels 必须为 2 (实部和虚部)。")

    real_part = data[:, 0, :]  # 提取实部
    imag_part = data[:, 1, :]  # 提取虚部
    if isinstance(real_part, np.ndarray):
        real_part = torch.tensor(real_part, dtype=torch.float32)
    if isinstance(imag_part, np.ndarray):
        imag_part = torch.tensor(imag_part, dtype=torch.float32)

    # 计算幅度
    magnitude = torch.sqrt(real_part ** 2 + imag_part ** 2)  # 幅度计算
    # 将幅度的维度调整为 (samples, 1, 8188)
    magnitude = magnitude.unsqueeze(1)  # 扩展为 (samples, 1, 8188)
    return magnitude
#---------------------------Compute phase---------------------------
def compute_phase(data):
    if data.shape[1] != 2:
        raise ValueError("输入数据的第二维 channels 必须为 2 (实部和虚部)。")

    real_part = data[:, 0, :]  # 提取实部
    imag_part = data[:, 1, :]  # 提取虚部
    # 计算相位
    phase = torch.atan2(imag_part, real_part)  # 相位计算
    # 将相位的维度调整为 (samples, 1, 8188)
    phase = phase.unsqueeze(1)  # 扩展为 (samples, 1, 8188)
    return phase


def save_data(train_data, test_data, train_labels, test_labels):
    data_output_folder = f'preprocess/processed_data/{args.user}_users/data'
    label_output_folder = f'preprocess/processed_data/{args.user}_users/labels'

    os.makedirs(data_output_folder, exist_ok=True)
    os.makedirs(label_output_folder, exist_ok=True)

    if args.channels == 2:
        # 保存数据
        np.save(os.path.join(data_output_folder, f'range{args.range}_channel{args.channels}_user{args.user}_{args.two_channel}_{args.data_split}_train_data.npy'), train_data.cpu().numpy())
        np.save(os.path.join(data_output_folder, f'range{args.range}_channel{args.channels}_user{args.user}_{args.two_channel}_{args.data_split}_test_data.npy'), test_data.cpu().numpy())

        # 保存标签
        np.save(os.path.join(label_output_folder, f'range{args.range}_channel{args.channels}_user{args.user}_{args.two_channel}_{args.data_split}_train_labels.npy'), train_labels.cpu().numpy())
        np.save(os.path.join(label_output_folder, f'range{args.range}_channel{args.channels}_user{args.user}_{args.two_channel}_{args.data_split}_test_labels.npy'), test_labels.cpu().numpy())

    elif args.channels == 3:
        # 保存数据
        np.save(os.path.join(data_output_folder, f'range{args.range}_channel{args.channels}_user{args.user}_{args.third_channel}_{args.data_split}_train_data.npy'), train_data.cpu().numpy())
        np.save(os.path.join(data_output_folder, f'range{args.range}_channel{args.channels}_user{args.user}_{args.third_channel}_{args.data_split}_test_data.npy'), test_data.cpu().numpy())

        # 保存标签
        np.save(os.path.join(label_output_folder, f'range{args.range}_channel{args.channels}_user{args.user}_{args.third_channel}_{args.data_split}_train_labels.npy'), train_labels.cpu().numpy())
        np.save(os.path.join(label_output_folder, f'range{args.range}_channel{args.channels}_user{args.user}_{args.third_channel}_{args.data_split}test_labels.npy'), test_labels.cpu().numpy())

    print("数据已保存到以下文件夹：")
    print(f"train_data: {train_data.shape}, train_labels: {train_labels.shape}")
    print(f"test_data: {test_data.shape}, test_labels: {test_labels.shape}")


def save_data_with_counter(prefix, user_id, train_data, test_data, train_folder, test_folder, args, counter):
    counter = 1
    while True:
        if args.channels == 2:
            train_filename = f"{prefix}_{args.range}_{args.data_split}_{args.normalization}_channel{args.channels}_{args.two_channel}_{counter}.npy"
            test_filename = f"{prefix}_{args.range}_{args.data_split}_{args.normalization}_channel{args.channels}_{args.two_channel}_{counter}.npy"
        elif args.channels == 3:
            train_filename = f"{prefix}_{args.range}_{args.data_split}_{args.normalization}_channel{args.channels}_{args.third_channel}_{counter}.npy"
            test_filename = f"{prefix}_{args.range}_{args.data_split}_{args.normalization}_channel{args.channels}_{args.third_channel}_{counter}.npy"

        train_file = os.path.join(train_folder, train_filename)
        test_file = os.path.join(test_folder, test_filename)

        if not os.path.exists(train_file) and not os.path.exists(test_file):
            break

        counter += 1

    np.save(train_file, train_data)  # 不用 .numpy()
    np.save(test_file, test_data)

    print(f"保存用户 {user_id} 的训练数据到: {train_file}")
    print(f"保存用户 {user_id} 的测试数据到: {test_file}")