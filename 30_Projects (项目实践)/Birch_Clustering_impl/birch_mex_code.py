import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import Birch
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, normalized_mutual_info_score, adjusted_rand_score
import glob
from tqdm import tqdm
import seaborn as sns
import warnings
import matplotlib
import gc

# 设置matplotlib支持中文显示
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决保存图像负号'-'显示为方块的问题

# 忽略警告
warnings.filterwarnings('ignore')


def ensure_directory(directory):
    """确保目录存在"""
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"创建目录: {directory}")


def load_merged_data(file_path):
    """加载合并后的传感器数据文件"""
    # 读取CSV文件
    df = pd.read_csv(file_path)

    # 从文件名提取元数据
    filename = os.path.basename(file_path)
    parts = filename.replace("unified_sensor_data_", "").replace(".csv", "").split("_")

    if len(parts) >= 3:
        subject_id = int(parts[0])
        action_id = int(parts[1])
        variant = int(parts[2])

        # 添加元数据列
        df['subject_id'] = subject_id
        df['action_id'] = action_id
        df['variant'] = variant

    return df


def preprocess_data(df):
    """预处理数据用于聚类"""
    # 排除非特征列
    non_feature_cols = ['timestamp', 'standard_timestamp', 'subject_id', 'action_id', 'variant']
    feature_cols = [col for col in df.columns if col not in non_feature_cols]

    print(f"使用 {len(feature_cols)} 个特征进行聚类")

    # 提取特征
    features = df[feature_cols].copy()

    # 处理缺失值
    features = features.fillna(features.mean())

    # 标准化特征
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    # 创建DataFrame以保留列名
    features_scaled = pd.DataFrame(scaled_features, columns=feature_cols)

    return features_scaled, scaler


def apply_birch_clustering(features, n_clusters=7, threshold=0.5, branching_factor=50):
    """
    应用BIRCH聚类算法，包含内存错误处理
    如果发生内存错误，会增加阈值并重试
    """
    print(f"应用BIRCH聚类 (n_clusters={n_clusters}, threshold={threshold})")

    while threshold <= 1.0:
        try:
            # 创建BIRCH模型
            birch = Birch(
                n_clusters=n_clusters,
                threshold=threshold,
                branching_factor=branching_factor
            )

            # 训练模型并获取聚类标签
            labels = birch.fit_predict(features)
            print(f"聚类成功完成，threshold={threshold}")
            return birch, labels

        except MemoryError:
            # 内存错误，增加阈值并重试
            gc.collect()  # 尝试释放内存
            threshold += 0.2
            print(f"发生内存错误！增加阈值重试，新threshold={threshold}")

        except Exception as e:
            # 其他错误
            print(f"聚类出错: {e}")
            threshold += 0.2
            print(f"增加阈值重试，新threshold={threshold}")

    # 如果所有尝试都失败
    print("所有threshold值都尝试失败，请检查数据或尝试其他参数")


def evaluate_clustering(features, labels, true_labels=None):
    """评估聚类结果"""
    metrics = {}

    # 计算聚类数量和分布
    unique_clusters = np.unique(labels)
    metrics['cluster_count'] = len(unique_clusters)
    cluster_counts = np.bincount(labels)
    metrics['cluster_distribution'] = cluster_counts

    print(f"生成了 {len(unique_clusters)} 个聚类")
    print("聚类分布:")
    for i, count in enumerate(cluster_counts):
        print(f"  聚类 {i}: {count} 样本 ({count / len(labels):.2%})")

    # 计算轮廓系数
    if len(unique_clusters) > 1:
        silhouette = silhouette_score(features, labels)
        metrics['silhouette_score'] = silhouette
        print(f"轮廓系数: {silhouette:.4f}")

    # 如果提供了真实标签，计算NMI和ARI
    if true_labels is not None:
        nmi = normalized_mutual_info_score(true_labels, labels)
        metrics['nmi_score'] = nmi
        print(f"标准化互信息 (NMI): {nmi:.4f}")

        ari = adjusted_rand_score(true_labels, labels)
        metrics['ari_score'] = ari
        print(f"调整兰德指数 (ARI): {ari:.4f}")

    return metrics


def analyze_action_cluster_relation(data, labels, output_path):
    """分析并可视化聚类和动作的关系"""
    # 创建临时DataFrame
    temp_df = pd.DataFrame({
        'cluster': labels,
        'action_id': data['action_id'].values
    })

    # 创建交叉表
    crosstab = pd.crosstab(
        temp_df['cluster'],
        temp_df['action_id'],
        normalize='index'
    )

    # 创建可视化
    plt.figure(figsize=(12, 9))
    sns.heatmap(crosstab, annot=True, cmap='YlGnBu', fmt='.2%')
    plt.title('聚类与动作类型的对应关系', fontsize=16)
    plt.xlabel('动作类型', fontsize=14)
    plt.ylabel('聚类编号', fontsize=14)
    plt.yticks(fontsize=12)
    plt.xticks(fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def load_all_data(input_dir, pattern="unified_sensor_data_*.csv"):
    """加载所有数据文件并合并成一个大的DataFrame"""
    print(f"正在加载所有数据文件 ({pattern}) 从目录: {input_dir}")

    # 查找所有匹配的文件
    file_paths = glob.glob(os.path.join(input_dir, pattern))
    print(f"找到 {len(file_paths)} 个文件")

    # 用于存储所有数据
    all_data = []

    # 加载每个文件
    for file_path in tqdm(file_paths, desc="加载数据文件"):
        df = load_merged_data(file_path)
        if df is not None and not df.empty:
            all_data.append(df)

    # 合并所有数据
    combined_data = pd.concat(all_data, ignore_index=True)
    print(f"合并完成，总数据量: {len(combined_data)} 行")

    return combined_data


def find_optimal_threshold(features, true_labels, metadata, n_clusters=8, threshold_range=[0.5, 0.7, 0.9]):
    """
    寻找BIRCH聚类的最佳阈值参数，使用分层采样以确保代表性
    簇的数量固定为8

    参数:
    features: 特征DataFrame
    true_labels: 真实标签
    metadata: 包含subject_id和action_id的DataFrame（用于分层采样）
    n_clusters: 固定的聚类数量，默认为8
    threshold_range: 阈值范围
    """
    print(f"正在寻找最佳BIRCH阈值参数（固定聚类数量为{n_clusters}）...")

    best_score = -1
    best_params = {}
    results = []

    # 对大型数据集进行分层采样以加速参数搜索
    if len(features) > 20000:
        sample_size = 20000

        # 创建分层采样的层（strata）
        # 使用action_id和subject_id的组合作为分层标准
        strata = metadata['action_id'].astype(str) + "_" + metadata['subject_id'].astype(str)

        # 计算每个层应该采样的数量（按比例）
        strata_counts = strata.value_counts()
        sampling_fractions = {}

        for stratum, count in strata_counts.items():
            # 计算采样比例 = 目标样本大小 / 总体大小
            fraction = min(1.0, sample_size * (count / len(features)))
            sampling_fractions[stratum] = fraction

        # 执行分层采样
        sampled_indices = []
        for stratum in strata_counts.index:
            # 获取该层的所有索引
            stratum_indices = strata[strata == stratum].index
            # 计算要采样的数量
            n_to_sample = int(len(stratum_indices) * sampling_fractions[stratum])
            # 确保至少采样1个点（如果该层存在）
            n_to_sample = max(1, min(n_to_sample, len(stratum_indices)))
            # 随机采样
            selected_indices = np.random.choice(stratum_indices, n_to_sample, replace=False)
            sampled_indices.extend(selected_indices)

        # 转换为array并确保不超过目标样本大小
        sampled_indices = np.array(sampled_indices)
        if len(sampled_indices) > sample_size:
            sampled_indices = np.random.choice(sampled_indices, sample_size, replace=False)

        # 基于采样索引获取样本
        features_sample = features.iloc[sampled_indices]
        true_labels_sample = true_labels[sampled_indices] if true_labels is not None else None

        print(f"使用分层采样获取 {len(features_sample)} 个样本进行参数优化")
        print(f"样本包含 {len(pd.Series(true_labels_sample).unique())} 种不同的动作类别")
    else:
        features_sample = features
        true_labels_sample = true_labels
        print("数据集较小，使用全部数据进行参数优化")

    # 仅遍历threshold_range，n_clusters固定为8
    for threshold in tqdm(threshold_range, desc="参数搜索"):
        try:
            # 应用BIRCH聚类
            birch = Birch(n_clusters=n_clusters, threshold=threshold)
            labels = birch.fit_predict(features_sample)

            # 计算轮廓系数
            unique_labels = np.unique(labels)
            if len(unique_labels) > 1:
                silhouette = silhouette_score(features_sample, labels)

                # 如果有真实标签，也计算NMI和ARI
                result = {
                    'n_clusters': n_clusters,
                    'threshold': threshold,
                    'silhouette_score': silhouette,
                    'unique_clusters': len(unique_labels)
                }

                if true_labels_sample is not None:
                    nmi = normalized_mutual_info_score(true_labels_sample, labels)
                    ari = adjusted_rand_score(true_labels_sample, labels)
                    result['nmi_score'] = nmi
                    result['ari_score'] = ari

                    # 使用NMI作为主要优化指标，而不是轮廓系数
                    score_to_optimize = nmi
                else:
                    score_to_optimize = silhouette

                results.append(result)

                # 更新最佳参数
                if score_to_optimize > best_score:
                    best_score = score_to_optimize
                    best_params = result.copy()
        except Exception as e:
            print(f"参数 n_clusters={n_clusters}, threshold={threshold} 出错: {e}")
            # 继续尝试下一个参数

    # 打印最佳参数
    if best_params:
        print("\n最佳参数:")
        for key, value in best_params.items():
            print(f"  {key}: {value}")
    else:
        # 如果没有找到最佳参数，返回默认值
        best_params = {'n_clusters': n_clusters, 'threshold': threshold_range[0]}
        print("\n未找到最佳参数，使用默认值:")
        print(f"  n_clusters: {n_clusters}")
        print(f"  threshold: {threshold_range[0]}")

    return best_params, pd.DataFrame(results)


def generate_report(all_data, labels, metrics, best_params, cluster_stats, output_path):
    """生成分析报告文本文件"""
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("MEx数据集BIRCH全局聚类分析报告\n")
        f.write("==========================\n\n")

        # 数据信息
        f.write("1. 数据集信息\n")
        f.write("--------------\n")
        f.write(f"总样本数: {len(all_data)}\n")
        f.write(f"特征数量: {len(all_data.columns) - 4}  # 排除metadata列\n\n")

        # 动作分布信息
        f.write("2. 动作类别分布\n")
        f.write("--------------\n")
        action_counts = all_data['action_id'].value_counts().sort_index()
        for action_id, count in action_counts.items():
            f.write(f"动作 {action_id}: {count} 样本 ({count / len(all_data):.2%})\n")
        f.write("\n")

        # 聚类参数
        f.write("3. 聚类参数\n")
        f.write("--------------\n")
        f.write(f"聚类算法: BIRCH\n")
        f.write(f"聚类数量: {best_params['n_clusters']}\n")
        f.write(f"阈值参数: {best_params['threshold']}\n\n")

        # 聚类结果
        f.write("4. 聚类结果\n")
        f.write("--------------\n")
        unique_clusters = np.unique(labels)
        f.write(f"生成聚类数: {len(unique_clusters)}\n")
        cluster_counts = np.bincount(labels)
        f.write("聚类分布:\n")
        for i, count in enumerate(cluster_counts):
            f.write(f"聚类 {i}: {count} 样本 ({count / len(labels):.2%})\n")
        f.write("\n")

        # 聚类与动作的对应关系
        f.write("5. 聚类与动作类型的对应关系\n")
        f.write("--------------\n")
        for i, stat in enumerate(cluster_stats):
            f.write(f"聚类 {i}: 主要对应动作 {stat['main_action']} ({stat['main_action_percent']:.2f}%)\n")
        f.write("\n")

        # 评估指标
        f.write("6. 聚类质量评估\n")
        f.write("--------------\n")
        f.write(f"轮廓系数 (Silhouette): {metrics.get('silhouette_score', 'N/A')}\n")
        f.write(f"标准化互信息 (NMI): {metrics.get('nmi_score', 'N/A')}\n")
        f.write(f"调整兰德指数 (ARI): {metrics.get('ari_score', 'N/A')}\n\n")

        f.write("7. 结论\n")
        f.write("--------------\n")
        f.write("根据NMI和ARI得分，聚类结果与真实的动作类别有一定的对应关系，但并不完全匹配。\n")
        if metrics.get('nmi_score', 0) > 0.3:
            f.write("NMI分数超过0.3，表明聚类结果捕捉到了部分动作类别的结构。\n")
        else:
            f.write("NMI分数较低，表明聚类结果与真实动作类别的对应关系较弱。\n")

        f.write("\n生成日期: 2023-05-23")


def perform_global_clustering(input_dir, output_dir):
    """对所有数据进行全局聚类"""
    print("\n执行全局聚类分析")
    print("================")

    # 创建全局聚类结果目录
    result_dir = os.path.join(output_dir, "global_clustering")
    ensure_directory(result_dir)

    # 加载所有数据
    all_data = load_all_data(input_dir)

    print(f"总数据量: {len(all_data)} 行")

    # 检查动作ID的分布
    action_counts = all_data['action_id'].value_counts().sort_index()
    print("\n动作类别分布:")
    for action_id, count in action_counts.items():
        print(f"  动作 {action_id}: {count} 样本 ({count / len(all_data):.2%})")

    # 数据预处理
    features, scaler = preprocess_data(all_data)

    # 创建元数据DataFrame，包含分层所需的列
    metadata = all_data[['subject_id', 'action_id']].copy()

    # 固定聚类数量为8，只寻找最佳的threshold值
    best_params, param_results = find_optimal_threshold(
        features,
        all_data['action_id'].values,
        metadata,  # 传入元数据用于分层采样
        n_clusters=8,  # 固定聚类数量为8
        threshold_range=[0.4, 0.5, 0.6, 0.7, 0.8, 0.9]  # 阈值范围
    )

    # 使用最佳的threshold值
    n_clusters = 8  # 固定为8
    threshold = best_params['threshold']

    print(f"\n使用参数: n_clusters={n_clusters}, threshold={threshold}")

    # 应用BIRCH聚类，包含内存错误处理
    birch, labels = apply_birch_clustering(features, n_clusters, threshold)

    # 评估聚类结果
    metrics = evaluate_clustering(features, labels, true_labels=all_data['action_id'].values)

    # 分析聚类与动作的关系
    analyze_action_cluster_relation(
        all_data,
        labels,
        os.path.join(result_dir, "cluster_action_relation.png")
    )

    # 保存带标签的数据
    all_data_with_labels = all_data.copy()
    all_data_with_labels['global_cluster'] = labels
    all_data_with_labels.to_csv(os.path.join(result_dir, "global_data_with_clusters.csv"), index=False)

    # 为每个聚类计算主要动作类型
    cluster_stats = []
    for cluster in range(max(labels) + 1):
        cluster_actions = all_data_with_labels[all_data_with_labels['global_cluster'] == cluster][
            'action_id'].value_counts()
        if not cluster_actions.empty:
            main_action = cluster_actions.idxmax()
            main_action_percent = cluster_actions.max() / cluster_actions.sum() * 100
            cluster_size = len(all_data_with_labels[all_data_with_labels['global_cluster'] == cluster])

            cluster_stats.append({
                'cluster': cluster,
                'size': cluster_size,
                'main_action': main_action,
                'main_action_percent': main_action_percent
            })

    # 生成分析报告文本文件
    generate_report(
        all_data,
        labels,
        metrics,
        {'n_clusters': n_clusters, 'threshold': threshold},
        cluster_stats,
        os.path.join(result_dir, "clustering_report.txt")
    )

    print(f"全局聚类结果已保存至目录: {result_dir}")
    return labels, metrics


if __name__ == "__main__":
    # 设置路径
    merged_dir = "D:/AfterDownload/Workspace/Python workspace/DataMining/mex_merged"
    output_dir = "D:/AfterDownload/Workspace/Python workspace/DataMining/mex_BIRCH"

    print("MEx数据集BIRCH全局聚类分析")
    print("========================")

    # 直接执行全局聚类分析
    perform_global_clustering(merged_dir, output_dir)