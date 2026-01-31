import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="CSI数据预处理脚本")
    parser.add_argument('--range', type=str,
                        choices=['300_aap1_55', '300_aap1_32', '300_aap1_82', '200_aap1_55', '100_aap1_55', '50_aap1_55',
                                 '300_aap2_55','300_aap2_82', '50_aap2_55', '300_aap2_32', '200_aap2_55', '100_aap2_55',
                                 '300_gburg_55', '300_gburg_82','300_gburg_32',
                                 '400_oats_55', '400_oats_82', '400_oats_32', '40_oats_55'],
                        default='400_oats', help='数据范围与场景选择')
    parser.add_argument('--channels', type=int, choices=[2, 3], default=2, help='通道数：2 或 3')
    parser.add_argument('--third_channel', type=str, default='all_magnitude', choices=['magnitude', 'phase', 'all_magnitude', 'all_phase'], help='第三通道选项')
    parser.add_argument('--two_channel', type=str, default='real_imag', choices=['real_imag', 'magnitude_phase'], help='二通道组成选项')

    parser.add_argument('--count', type=int, help='用户对数量')
    parser.add_argument('--type', type=str, default='cir', choices=['cir', 'cfr'])
    parser.add_argument('--normalization', type=str, choices=['none', 'minmax', 'standardize'], default='none',
                        help="选择归一化方法")
    parser.add_argument('--data_split', default='4_1', type=str, choices=['random_82', 'random_55', 'random_28', 'sequence','2_1','3_1', '4_1', '5_1', '6_1'])


    return parser.parse_args()