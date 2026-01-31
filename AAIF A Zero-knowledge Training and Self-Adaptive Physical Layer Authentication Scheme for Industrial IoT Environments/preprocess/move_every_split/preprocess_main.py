import numpy as np
import h5py
import os
import torch
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from parser import parse_args
from user_range import user_ranges_dict
from utils import (
    normalize_with_minmax_per_channel,
    standardize_per_channel,
    compute_magnitude,
    compute_phase,
    save_data_with_counter
)
from scipy.signal import savgol_filter


def split_indices(num_samples, data_split):
    if args.data_split == 'random_82':
        indices = np.arange(num_samples)
        np.random.shuffle(indices)
        train_size = int(num_samples * 0.8)
        train_indices = indices[:train_size]
        test_indices = indices[train_size:]
    elif args.data_split == 'random_55':
        indices = np.arange(num_samples)
        np.random.shuffle(indices)
        train_size = int(num_samples * 0.5)
        train_indices = indices[:train_size]
        test_indices = indices[train_size:]
    elif args.data_split == 'random_28':
        indices = np.arange(num_samples)
        np.random.shuffle(indices)
        train_size = int(num_samples * 0.2)
        train_indices = indices[:train_size]
        test_indices = indices[train_size:]
    elif data_split == 'sequence':
        train_size = int(num_samples * 0.8)
        train_indices = np.arange(train_size)
        test_indices = np.arange(train_size, num_samples)
    elif data_split == '2_1':
        train_indices = [i for i in range(num_samples) if i % 3 != 2]
        test_indices = [i for i in range(num_samples) if i % 3 == 2]
    elif data_split == '3_1':
        train_indices = [i for i in range(num_samples) if i % 4 != 3]
        test_indices = [i for i in range(num_samples) if i % 4 == 3]
    elif data_split == '4_1':
        train_indices = [i for i in range(num_samples) if i % 5 != 4]
        test_indices = [i for i in range(num_samples) if i % 5 == 4]
    elif data_split == '5_1':
        train_indices = [i for i in range(num_samples) if i % 6 != 5]
        test_indices = [i for i in range(num_samples) if i % 6 == 5]
    else:
        raise ValueError(f"Unsupported data_split strategy: {data_split}")
    return train_indices, test_indices


def process_user_data(user_id, start, end, real_part, imag_part, args):
    num_samples = end - start + 1
    print(f"User {user_id} 的样本数量: {num_samples}")

    current_real = real_part[start:end + 1]
    current_imag = imag_part[start:end + 1]

    train_idx, test_idx = split_indices(num_samples, args.data_split)
    train_real = current_real[train_idx]
    test_real = current_real[test_idx]
    train_imag = current_imag[train_idx]
    test_imag = current_imag[test_idx]

    train_data = np.concatenate((train_real[:, np.newaxis, :], train_imag[:, np.newaxis, :]), axis=1)
    test_data = np.concatenate((test_real[:, np.newaxis, :], test_imag[:, np.newaxis, :]), axis=1)

    dummy_labels = np.full(train_data.shape[0], user_id)

    train_data, _ = shuffle(train_data, dummy_labels, random_state=42)
    test_dummy = np.full(test_data.shape[0], user_id)
    test_data, _ = shuffle(test_data, test_dummy, random_state=42)

    if args.normalization == 'minmax':
        if args.channels == 2:
            if args.two_channel == 'real_imag':
                train_min = []
                train_max = []
                for c in range(train_data.shape[1]):
                    train_min.append(train_data[:, c, :].min())
                    train_max.append(train_data[:, c, :].max())
                train_min = np.array(train_min)
                train_max = np.array(train_max)
                train_data = normalize_with_minmax_per_channel(train_data, train_min, train_max)
                test_data = normalize_with_minmax_per_channel(test_data, train_min, train_max)

            elif args.two_channel == 'magnitude_phase':

                magnitude_temp_train = compute_magnitude(train_data)
                magnitude_temp_test = compute_magnitude(test_data)

                phase_temp_train = compute_magnitude(train_data)
                phase_temp_test = compute_magnitude(test_data)

                train_data = np.concatenate((magnitude_temp_train, phase_temp_train), axis=1)
                test_data = np.concatenate((magnitude_temp_test, phase_temp_test), axis=1)

                train_min = []
                train_max = []
                for c in range(train_data.shape[1]):
                    train_min.append(train_data[:, c, :].min())
                    train_max.append(train_data[:, c, :].max())
                train_min = np.array(train_min)
                train_max = np.array(train_max)
                train_data = normalize_with_minmax_per_channel(train_data, train_min, train_max)
                test_data = normalize_with_minmax_per_channel(test_data, train_min, train_max)

        elif args.channels == 3:
            if args.third_channel == 'magnitude':
                print("使用 min-max 归一化，并增加幅度通道...")
                temp_train = compute_magnitude(train_data)
                temp_test = compute_magnitude(test_data)
                train_data = np.concatenate((train_data, temp_train), axis=1)
                test_data = np.concatenate((test_data, temp_test), axis=1)
                train_min = []
                train_max = []
                for c in range(train_data.shape[1]):
                    train_min.append(train_data[:, c, :].min())
                    train_max.append(train_data[:, c, :].max())
                train_min = np.array(train_min)
                train_max = np.array(train_max)
                train_data = normalize_with_minmax_per_channel(train_data, train_min, train_max)
                test_data = normalize_with_minmax_per_channel(test_data, train_min, train_max)
            elif args.third_channel == 'phase':
                print("使用 min-max 归一化，并增加相位通道...")
                temp_train = compute_phase(train_data)
                temp_test = compute_phase(test_data)
                train_data = np.concatenate((train_data, temp_train), axis=1)
                test_data = np.concatenate((test_data, temp_test), axis=1)
                train_min = []
                train_max = []
                for c in range(train_data.shape[1]):
                    train_min.append(train_data[:, c, :].min())
                    train_max.append(train_data[:, c, :].max())
                train_min = np.array(train_min)
                train_max = np.array(train_max)
                train_data = normalize_with_minmax_per_channel(train_data, train_min, train_max)
                test_data = normalize_with_minmax_per_channel(test_data, train_min, train_max)
            elif args.third_channel == 'all_magnitude':
                print("使用 min-max 归一化：所有通道均为幅度信息...")
                temp_train = compute_magnitude(train_data)
                temp_test = compute_magnitude(test_data)
                train_data = np.concatenate([temp_train, temp_train, temp_train], axis=1)
                test_data = np.concatenate([temp_test, temp_test, temp_test], axis=1)
                train_min = []
                train_max = []
                for c in range(train_data.shape[1]):
                    train_min.append(train_data[:, c, :].min())
                    train_max.append(train_data[:, c, :].max())
                train_min = np.array(train_min)
                train_max = np.array(train_max)
                train_data = normalize_with_minmax_per_channel(train_data, train_min, train_max)
                test_data = normalize_with_minmax_per_channel(test_data, train_min, train_max)
            elif args.third_channel == 'all_phase':
                print("使用 min-max 归一化：所有通道均为相位信息...")
                temp_train = compute_phase(train_data)
                temp_test = compute_phase(test_data)
                train_data = np.concatenate([temp_train, temp_train, temp_train], axis=1)
                test_data = np.concatenate([temp_test, temp_test, temp_test], axis=1)
                train_min = []
                train_max = []
                for c in range(train_data.shape[1]):
                    train_min.append(train_data[:, c, :].min())
                    train_max.append(train_data[:, c, :].max())
                train_min = np.array(train_min)
                train_max = np.array(train_max)
                train_data = normalize_with_minmax_per_channel(train_data, train_min, train_max)
                test_data = normalize_with_minmax_per_channel(test_data, train_min, train_max)

    else:
        if args.channels == 2:
            if args.two_channel == 'real_imag':
                print("Directly process......")
            elif args.two_channel == 'magnitude_phase':

                magnitude_temp_train = compute_magnitude(train_data)
                magnitude_temp_test = compute_magnitude(test_data)

                phase_temp_train = compute_magnitude(train_data)
                phase_temp_test = compute_magnitude(test_data)

                train_data = np.concatenate((magnitude_temp_train, phase_temp_train), axis=1)
                test_data = np.concatenate((magnitude_temp_test, phase_temp_test), axis=1)

        elif args.channels == 3:
            if args.third_channel == 'magnitude':
                temp1 = compute_magnitude(train_data)
                temp2 = compute_magnitude(test_data)

                print('Shape of temp1&temp2 is ', temp1.shape, temp2.shape)

                train_data = np.concatenate((train_data, temp1), axis=1)
                test_data = np.concatenate((test_data, temp2), axis=1)

                if np.allclose(temp1[:, 0, :], np.sqrt(train_data[:, 0, :] ** 2 + train_data[:, 1, :] ** 2), atol=1e-9):
                    print("Meet the requirement!")
                else:
                    print("Does not meet the requirement!")

                train_data, test_data = torch.tensor(train_data), torch.tensor(test_data)

                print('Shape of train_data is ', train_data.shape)
                print('Shape of test_data is ', test_data.shape)

            elif args.third_channel == 'phase':
                temp1 = compute_phase(train_data)
                temp2 = compute_phase(test_data)

                print('Shape of temp1&temp2 is ', temp1.shape, temp2.shape)

                train_data = np.concatenate((train_data, temp1), axis=1)
                test_data = np.concatenate((test_data, temp2), axis=1)

                train_data, test_data = torch.tensor(train_data), torch.tensor(test_data)

                print('Shape of train_data is ', train_data.shape)
                print('Shape of test_data is ', test_data.shape)

            elif args.third_channel == 'all_magnitude':
                temp1 = compute_magnitude(train_data)
                temp2 = compute_magnitude(test_data)
                print('Shape of temp1:', temp1.shape, 'Type:', type(temp1))
                print('Shape of temp2:', temp2.shape, 'Type:', type(temp2))
                train_data = np.concatenate([temp1, temp1, temp1], axis=1)
                test_data = np.concatenate([temp2, temp2, temp2], axis=1)

                train_data, test_data = torch.tensor(train_data), torch.tensor(test_data)

                print('Shape of train_data is ', train_data.shape)
                print('Shape of test_data is ', test_data.shape)

            elif args.third_channel == 'all_phase':
                temp1 = compute_phase(train_data)
                temp2 = compute_phase(test_data)
                print('Shape of temp1:', temp1.shape, 'Type:', type(temp1))
                print('Shape of temp2:', temp2.shape, 'Type:', type(temp2))
                train_data = np.concatenate([temp1, temp1, temp1], axis=1)
                test_data = np.concatenate([temp2, temp2, temp2], axis=1)

                train_data, test_data = torch.tensor(train_data), torch.tensor(test_data)

                print('Shape of train_data is ', train_data.shape)
                print('Shape of test_data is ', test_data.shape)

    train_data = torch.from_numpy(np.array(train_data)).float() if not isinstance(train_data,
                                                                                  torch.Tensor) else train_data.float()
    test_data = torch.from_numpy(np.array(test_data)).float() if not isinstance(test_data,
                                                                                torch.Tensor) else test_data.float()

    return train_data, test_data


def preprocess_CSI_data(args):
    if args.type == 'cfr':
        if args.range in ['400_oats_82', '400_oats_55', '400_oats_32', '40_oats_55']:
            input_file = 'preprocess/data/CFR/Oats_5G_3115horn_vpol_run36b_pp.mat'
            scenario='OATS'
        elif args.range in ['300_aap1_82', '300_aap1_55', '300_aap1_32', '50_aap1_55', '200_aap1_55', '100_aap1_55',]:
            input_file = 'preprocess/data/CFR/AAPlantD2_5GHz_TX1_vpol_run1_pp.mat'
            scenario = 'AAP1'
        elif args.range in ['300_aap2_55', '300_aap2_82', '300_aap2_32', '50_aap2_55', '200_aap2_55', '100_aap2_55',]:
            input_file = 'preprocess/data/CFR/AAPlantD3_5GHz_TX2b_vpol_internal_run40_pp.mat'
            scenario = 'AAP2'
        elif args.range in ['300_gburg_55', '300_gburg_82', '300_gburg_32']:
            input_file = 'preprocess/data/CFR/GBurgD2_5GHz_TX1_vpol_run2_pp_reshape.mat'
            scenario = 'GBurg'
        else:
            raise ValueError("未知的 range 参数，请检查！")
    elif args.type == 'cir':
        if args.range in ['400_oats_82', '400_oats_55', '400_oats_32', '40_oats_55']:
            input_file = 'preprocess/data/CIR/Oats_5G_3115horn_vpol_run36b_pp.mat'
            scenario='OATS'
        elif args.range in ['300_aap1_82', '300_aap1_55', '300_aap1_32', '50_aap1_55', '200_aap1_55', '100_aap1_55',]:
            input_file = 'preprocess/data/CIR/AAPlantD2_5GHz_TX1_vpol_run1_pp.mat'
            scenario = 'AAP1'
        elif args.range in ['300_aap2_55', '300_aap2_82', '300_aap2_32', '50_aap2_55', '200_aap2_55', '100_aap2_55',]:
            input_file = 'preprocess/data/CIR/AAPlantD3_5GHz_TX2b_vpol_internal_run40_pp.mat'
            scenario = 'AAP2'
        elif args.range in ['300_gburg_55', '300_gburg_82', '300_gburg_32']:
            input_file = 'preprocess/data/CIR/GBurgD2_5GHz_TX1_vpol_run2_pp_reshape.mat'
            scenario = 'GBurg'
        else:
            raise ValueError("未知的 range 参数，请检查！")
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"文件 {input_file} 不存在，请检查路径！")
    print("路径是否存在：", os.path.exists(input_file))
    print("文件大小是否为0：", os.path.getsize(input_file))
    print(f"input_file = {input_file}")
    with h5py.File(input_file, 'r') as f:
        IQdata = f['IQdata'][:]
        if IQdata.dtype.names is not None:
            IQdata = IQdata['real'] + 1j * IQdata['imag']

    IQdata = IQdata.astype(np.complex64)
    original_length, num_samples_total = IQdata.shape
    print(f"原始数据维度：{original_length} x {num_samples_total}")

    if hasattr(args, 'window_length') and args.window_length is not None and \
            hasattr(args, 'polyorder') and args.polyorder is not None:
        window_length = args.window_length
        polyorder = args.polyorder
        if window_length % 2 == 0:
            window_length += 1
        if polyorder >= window_length:
            polyorder = window_length - 1
        print(f"应用 Savitzky-Golay 滤波: window_length={window_length}, polyorder={polyorder}")
        real_part = np.real(IQdata)
        imag_part = np.imag(IQdata)
        real_part = savgol_filter(real_part, window_length=window_length, polyorder=polyorder, axis=1)
        imag_part = savgol_filter(imag_part, window_length=window_length, polyorder=polyorder, axis=1)
        IQdata = real_part + 1j * imag_part
    else:
        print("未传入 window_length 或 polyorder 参数，跳过 Savitzky-Golay 滤波。")
        real_part = np.real(IQdata)
        imag_part = np.imag(IQdata)

    train_folder = os.path.join(f'./preprocess/processed_data/{scenario}/{args.count}', 'train')
    test_folder = os.path.join(f'./preprocess/processed_data/{scenario}/{args.count}', 'test')
    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(test_folder, exist_ok=True)

    for counter in range(1, args.count + 1):
        if args.range in user_ranges_dict:
            current_ranges = user_ranges_dict[args.range].get(counter, [])
            if not current_ranges:
                print(f"No user ranges for counter {counter}, skipping.")
                continue

            # ---------- 用户0（单段） ----------
            user0_range = current_ranges[0]  # [start, end]
            train_data, test_data = process_user_data(
                0, user0_range[0], user0_range[1], real_part, imag_part, args
            )
            save_data_with_counter(
                "legal", 0, train_data, test_data, train_folder, test_folder, args, counter
            )

            # ---------- 用户1（单段或多段） ----------
            user1_ranges = current_ranges[1]  # 可能是单段 [start, end] 或多段 [[s1,e1],[s2,e2],...]

            # 判断类型：如果是单段就直接处理
            if isinstance(user1_ranges[0], int):
                start, end = user1_ranges
                train_data, test_data = process_user_data(
                    1, start, end, real_part, imag_part, args
                )
            else:
                # 多段情况，堆叠所有段的数据
                all_train_data = []
                all_test_data = []
                for segment in user1_ranges:
                    start, end = segment
                    t_data, te_data = process_user_data(1, start, end, real_part, imag_part, args)
                    all_train_data.append(t_data)
                    all_test_data.append(te_data)
                train_data = np.concatenate(all_train_data, axis=0)
                test_data = np.concatenate(all_test_data, axis=0)

            save_data_with_counter(
                "illegal", 1, train_data, test_data, train_folder, test_folder, args, counter
            )

        else:
            print(f"No user ranges for range type {args.range}, skipping.")


if __name__ == "__main__":
    args = parse_args()
    preprocess_CSI_data(args)