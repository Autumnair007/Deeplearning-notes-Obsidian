#!/usr/bin/env python3
"""
CLIP模型训练和测试的简单示例
使用方法：
1. 正常训练: python test_clip.py
2. 仅测试: python test_clip.py --test_only
3. 强制重新训练: python test_clip.py --force_train
"""

import os
import sys

print("CLIP模型训练测试")
print("=" * 50)

# 检查数据集
current_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(current_dir, "sample_data")

print(f"当前目录: {current_dir}")
print(f"数据集路径: {data_path}")
print(f"数据集存在: {os.path.exists(data_path)}")

# 检查模型检查点
checkpoints_dir = os.path.join(current_dir, "checkpoints")
print(f"检查点目录: {checkpoints_dir}")
print(f"检查点目录存在: {os.path.exists(checkpoints_dir)}")

if os.path.exists(checkpoints_dir):
    checkpoint_files = [f for f in os.listdir(checkpoints_dir) if f.endswith('.pt')]
    print(f"已有检查点文件: {checkpoint_files}")
else:
    print("未找到检查点文件")

print("\n可用的命令行参数:")
print("  --test_only     : 仅测试模式（需要已有模型）")
print("  --force_train   : 强制重新训练")
print("  --epochs 5      : 设置训练轮数")
print("  --batch_size 16 : 设置批次大小")

print("\n示例命令:")
print("  python clip_train.py --epochs 3 --batch_size 16")
print("  python clip_train.py --test_only")
print("  python clip_train.py --force_train --epochs 5")
