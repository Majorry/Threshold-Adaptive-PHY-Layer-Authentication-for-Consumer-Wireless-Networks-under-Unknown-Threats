# -*- coding: gbk -*-
import numpy as np
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
    """重塑数据形状为 (样本数, 通道数, 长度)"""
    return data.reshape(data.shape[0], 2, 8188)