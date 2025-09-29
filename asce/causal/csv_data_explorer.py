#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CSV数据探索工具
用于查看和分析CSV文件中的数据，特别是用于因果分析的数据
"""

import pandas as pd
import numpy as np
import os
import argparse
import matplotlib.pyplot as plt
import seaborn as sns

def explore_csv_data(csv_path, user_id=None, output_dir=None):
    """
    探索CSV文件中的数据
    
    参数:
        csv_path: CSV文件路径
        user_id: 用户ID（如果指定，则只分析该用户的数据）
        output_dir: 输出目录（如果指定，则保存分析结果）
    """
    # 检查文件是否存在
    if not os.path.exists(csv_path):
        print(f"错误: CSV文件不存在: {csv_path}")
        return
    
    # 读取CSV文件
    try:
        df = pd.read_csv(csv_path)
        print(f"成功读取CSV文件: {csv_path}, 共 {len(df)} 行")
    except Exception as e:
        print(f"读取CSV文件时出错: {e}")
        return
    
    # 显示数据基本信息
    print("\n=== 数据基本信息 ===")
    print(f"列名: {df.columns.tolist()}")
    print(f"数据类型:\n{df.dtypes}")
    print(f"缺失值统计:\n{df.isnull().sum()}")
    
    # 显示数据样例
    print("\n=== 数据样例 ===")
    print(df.head())
    
    # 统计用户数量和每个用户的数据量
    if 'user_id' in df.columns:
        user_counts = df['user_id'].value_counts()
        print(f"\n=== 用户数据统计 ===")
        print(f"总用户数: {len(user_counts)}")
        print(f"每个用户的数据量统计:")
        print(f"最大值: {user_counts.max()}")
        print(f"最小值: {user_counts.min()}")
        print(f"平均值: {user_counts.mean():.2f}")
        print(f"中位数: {user_counts.median()}")
        
        # 显示数据量最多的前10个用户
        print(f"\n数据量最多的前10个用户:")
        print(user_counts.head(10))
        
        # 显示数据量最少的前10个用户
        print(f"\n数据量最少的前10个用户:")
        print(user_counts.tail(10))
    
    # 如果指定了用户ID，则只分析该用户的数据
    if user_id is not None and 'user_id' in df.columns:
        user_df = df[df['user_id'] == user_id]
        if user_df.empty:
            print(f"\n用户 {user_id} 在数据中不存在")
            return
        
        print(f"\n=== 用户 {user_id} 的数据分析 ===")
        print(f"数据量: {len(user_df)}")
        
        # 显示该用户的所有数据
        print("\n用户数据:")
        print(user_df)
        
        # 分析认知状态数据
        cognitive_cols = [col for col in user_df.columns if any(x in col for x in ['mood', 'emotion', 'thinking', 'stance', 'intention'])]
        if cognitive_cols:
            print(f"\n认知状态数据列: {cognitive_cols}")
            print("\n认知状态数据统计:")
            print(user_df[cognitive_cols].describe())
            
            # 检查认知状态数据的类型
            print("\n认知状态数据类型:")
            for col in cognitive_cols:
                print(f"{col}: {user_df[col].unique()}")
        
        # 如果有timestep列，则按时间顺序排序并显示
        if 'timestep' in user_df.columns:
            print("\n按时间顺序排序的数据:")
            print(user_df.sort_values('timestep'))
    
    # 如果指定了输出目录，则保存分析结果
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存数据基本信息
        with open(os.path.join(output_dir, 'data_info.txt'), 'w') as f:
            f.write(f"CSV文件: {csv_path}\n")
            f.write(f"总行数: {len(df)}\n")
            f.write(f"列名: {df.columns.tolist()}\n")
            f.write(f"数据类型:\n{df.dtypes}\n")
            f.write(f"缺失值统计:\n{df.isnull().sum()}\n")
            
            if 'user_id' in df.columns:
                f.write(f"\n用户数据统计:\n")
                f.write(f"总用户数: {len(user_counts)}\n")
                f.write(f"每个用户的数据量统计:\n")
                f.write(f"最大值: {user_counts.max()}\n")
                f.write(f"最小值: {user_counts.min()}\n")
                f.write(f"平均值: {user_counts.mean():.2f}\n")
                f.write(f"中位数: {user_counts.median()}\n")
        
        print(f"\n分析结果已保存至: {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CSV数据探索工具")
    parser.add_argument("--csv_path", type=str, required=True, help="CSV文件路径")
    parser.add_argument("--user_id", type=int, help="用户ID（如果指定，则只分析该用户的数据）")
    parser.add_argument("--output_dir", type=str, help="输出目录（如果指定，则保存分析结果）")
    
    args = parser.parse_args()
    
    explore_csv_data(args.csv_path, args.user_id, args.output_dir)
