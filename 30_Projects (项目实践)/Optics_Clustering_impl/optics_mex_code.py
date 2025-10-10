import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import OPTICS
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, normalized_mutual_info_score, adjusted_rand_score
import glob
from tqdm import tqdm
import seaborn as sns
import warnings
from datetime import datetime
import sys

warnings.filterwarnings('ignore')

# Windows中文字体设置
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


class Logger:
    """日志记录器"""

    def __init__(self, log_file):
        self.terminal = sys.stdout
        self.log = open(log_file, 'w', encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        pass

    def close(self):
        self.log.close()


def load_all_data(input_dir):
    """加载所有传感器数据文件"""
    file_paths = glob.glob(os.path.join(input_dir, "unified_sensor_data_*.csv"))
    print(f"找到 {len(file_paths)} 个文件")

    all_data = []
    for file_path in tqdm(file_paths, desc="加载数据"):
        df = pd.read_csv(file_path)
        # 从文件名解析信息
        filename = os.path.basename(file_path)
        parts = filename.replace("unified_sensor_data_", "").replace(".csv", "").split("_")
        if len(parts) >= 3:
            df['subject_id'] = int(parts[0])
            df['action_id'] = int(parts[1])
            df['variant'] = int(parts[2])
        all_data.append(df)

    combined_data = pd.concat(all_data, ignore_index=True)
    print(f"总数据量: {len(combined_data)} 行")
    return combined_data


def preprocess_data(df):
    """数据预处理"""
    non_feature_cols = ['timestamp', 'standard_timestamp', 'subject_id', 'action_id', 'variant']
    feature_cols = [col for col in df.columns if col not in non_feature_cols]
    print(f"使用 {len(feature_cols)} 个特征进行聚类")

    features = df[feature_cols].fillna(df[feature_cols].mean())
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    return pd.DataFrame(scaled_features, columns=feature_cols)


def stratified_sampling(data, sample_size=8000):
    """分层采样"""
    print(f"分层采样，目标: {sample_size} 样本")
    strata = data.groupby(['action_id', 'subject_id']).size().reset_index(name='count')
    total_samples = len(data)
    strata['target_samples'] = (strata['count'] / total_samples * sample_size).round().astype(int)
    strata['target_samples'] = strata['target_samples'].clip(lower=1)

    if strata['target_samples'].sum() > sample_size:
        scale_factor = sample_size / strata['target_samples'].sum()
        strata['target_samples'] = (strata['target_samples'] * scale_factor).round().astype(int)
        strata['target_samples'] = strata['target_samples'].clip(lower=1)

    sampled_data = []
    for _, stratum in strata.iterrows():
        stratum_data = data[(data['action_id'] == stratum['action_id']) &
                            (data['subject_id'] == stratum['subject_id'])]
        if len(stratum_data) <= stratum['target_samples']:
            sampled_data.append(stratum_data)
        else:
            sampled_data.append(stratum_data.sample(n=stratum['target_samples'], random_state=42))

    result = pd.concat(sampled_data, ignore_index=True)
    print(f"采样完成: {len(result)} 样本")
    return result


def grid_search_optics(features, true_labels):
    """OPTICS网格搜索"""
    print("开始OPTICS网格搜索（基于NMI优化）")
    param_grid = {
        'min_samples': [5, 10, 20],
        'xi': [0.01, 0.05, 0.1],
        'min_cluster_size': [10, 20, 50]
    }

    best_score = -1
    best_params = None
    results = []
    total_combinations = len(param_grid['min_samples']) * len(param_grid['xi']) * len(param_grid['min_cluster_size'])

    with tqdm(total=total_combinations, desc="网格搜索进度") as pbar:
        for min_samples in param_grid['min_samples']:
            for xi in param_grid['xi']:
                for min_cluster_size in param_grid['min_cluster_size']:
                    optics = OPTICS(min_samples=min_samples, xi=xi,
                                    min_cluster_size=min_cluster_size,
                                    cluster_method='xi', metric='euclidean', n_jobs=-1)
                    labels = optics.fit_predict(features)

                    unique_clusters = np.unique(labels[labels != -1])
                    n_clusters = len(unique_clusters)
                    n_noise = np.sum(labels == -1)

                    # 计算轮廓系数
                    if n_clusters > 1 and len(labels[labels != -1]) > 1:
                        non_noise_mask = labels != -1
                        if np.sum(non_noise_mask) > 3000:
                            non_noise_indices = np.where(non_noise_mask)[0]
                            idx = np.random.choice(non_noise_indices, 3000, replace=False)
                            silhouette = silhouette_score(features.iloc[idx], labels[idx])
                        else:
                            silhouette = silhouette_score(features.iloc[non_noise_mask], labels[non_noise_mask])
                    else:
                        silhouette = -1

                    nmi = normalized_mutual_info_score(true_labels, labels)
                    ari = adjusted_rand_score(true_labels, labels)

                    results.append({
                        'min_samples': min_samples, 'xi': xi, 'min_cluster_size': min_cluster_size,
                        'n_clusters': n_clusters, 'n_noise': n_noise,
                        'silhouette_score': round(silhouette, 8),
                        'nmi_score': round(nmi, 8),
                        'ari_score': round(ari, 8)
                    })

                    if nmi > best_score:
                        best_score = nmi
                        best_params = {'min_samples': min_samples, 'xi': xi, 'min_cluster_size': min_cluster_size}

                    print(f"min_samples={min_samples}, xi={xi}, min_cluster_size={min_cluster_size}: "
                          f"clusters={n_clusters}, noise={n_noise}, nmi={nmi:.8f}")

                    pbar.update(1)

    return best_params, pd.DataFrame(results)


def evaluate_clustering(features, labels, true_labels):
    """评估聚类结果"""
    unique_clusters = np.unique(labels[labels != -1])
    n_noise = np.sum(labels == -1)

    print(f"聚类数量: {len(unique_clusters)}")
    print(f"噪声点数量: {n_noise}")

    # 计算轮廓系数
    if len(unique_clusters) > 1 and len(labels[labels != -1]) > 1:
        non_noise_mask = labels != -1
        if np.sum(non_noise_mask) > 10000:
            non_noise_indices = np.where(non_noise_mask)[0]
            idx = np.random.choice(non_noise_indices, 10000, replace=False)
            silhouette = silhouette_score(features.iloc[idx], labels[idx])
        else:
            silhouette = silhouette_score(features.iloc[non_noise_mask], labels[non_noise_mask])
    else:
        silhouette = 0

    nmi = normalized_mutual_info_score(true_labels, labels)
    ari = adjusted_rand_score(true_labels, labels)

    silhouette = round(silhouette, 8)
    nmi = round(nmi, 8)
    ari = round(ari, 8)

    print(f"轮廓系数: {silhouette:.8f}")
    print(f"NMI: {nmi:.8f}")
    print(f"ARI: {ari:.8f}")

    return {'silhouette_score': silhouette, 'nmi_score': nmi, 'ari_score': ari}


def create_heatmap(data, labels, output_path, top_n_clusters=10):
    """创建聚类热力图"""
    non_noise_mask = labels != -1
    filtered_labels = labels[non_noise_mask]
    filtered_data = data[non_noise_mask]

    if len(filtered_labels) == 0:
        print("没有足够的非噪声聚类数据生成热力图")
        return

    temp_df = pd.DataFrame({'cluster': filtered_labels, 'action_id': filtered_data['action_id'].values})

    # 选择前N个最大聚类
    label_counts = pd.Series(filtered_labels).value_counts()
    top_clusters = label_counts.head(top_n_clusters).index.tolist()
    temp_df = temp_df[temp_df['cluster'].isin(top_clusters)]

    # 创建交叉表
    crosstab = pd.crosstab(temp_df['cluster'], temp_df['action_id'], normalize='index')

    # 确保包含所有动作ID
    all_action_ids = sorted(filtered_data['action_id'].unique())
    for action_id in all_action_ids:
        if action_id not in crosstab.columns:
            crosstab[action_id] = 0

    crosstab = crosstab.reindex(columns=sorted(crosstab.columns))

    plt.figure(figsize=(14, 10))
    sns.heatmap(crosstab, annot=True, cmap='Blues', fmt='.4f',
                cbar_kws={'label': '比例'}, linewidths=0.5)
    plt.title(f'OPTICS前{top_n_clusters}个聚类与7个动作类型的对应关系', fontsize=16)
    plt.xlabel('动作类型', fontsize=14)
    plt.ylabel('聚类编号', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"热力图已保存: {output_path}")


def create_reachability_plot(optics_model, output_path):
    """创建OPTICS可达性图"""
    plt.figure(figsize=(15, 8))

    # 绘制可达性距离图
    reachability = optics_model.reachability_[optics_model.ordering_]
    plt.plot(reachability, color='black', linewidth=1)
    plt.fill_between(range(len(reachability)), reachability, alpha=0.3, color='lightblue')

    plt.title('OPTICS可达性图', fontsize=16)
    plt.xlabel('数据点（按聚类顺序）', fontsize=14)
    plt.ylabel('可达性距离', fontsize=14)
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"可达性图已保存: {output_path}")


def main():
    """主函数"""
    # 路径设置
    merged_dir = "D:/AfterDownload/Workspace/Python workspace/DataMining/mex_merged"
    output_dir = "D:/AfterDownload/Workspace/Python workspace/DataMining/mex_OPTICS_GridSearch"

    os.makedirs(output_dir, exist_ok=True)

    # 设置日志记录
    log_file = os.path.join(output_dir, f"log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
    logger = Logger(log_file)
    sys.stdout = logger

    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"用户: Autumnair007")
    print("=== OPTICS聚类算法 ===")

    # 1. 加载数据
    print("\n=== 步骤1: 加载数据 ===")
    full_data = load_all_data(merged_dir)

    # 2. 分层采样
    print("\n=== 步骤2: 分层采样 ===")
    sampled_data = stratified_sampling(full_data, sample_size=8000)
    features_gs = preprocess_data(sampled_data)
    true_labels_gs = sampled_data['action_id'].values

    # 3. 网格搜索
    print("\n=== 步骤3: 网格搜索 ===")
    best_params, results_df = grid_search_optics(features_gs, true_labels_gs)
    results_df.to_csv(os.path.join(output_dir, 'optics_grid_search_results.csv'),
                      index=False, float_format='%.8f')
    print(f"最佳参数: {best_params}")

    # 4. 完整数据预处理
    print("\n=== 步骤4: 预处理完整数据 ===")
    features_full = preprocess_data(full_data)
    true_labels_full = full_data['action_id'].values

    # 5. 使用最佳参数进行聚类
    print("\n=== 步骤5: 完整数据聚类 ===")
    optics_final = OPTICS(**best_params, cluster_method='xi', metric='euclidean', n_jobs=-1)
    labels_full = optics_final.fit_predict(features_full)

    # 6. 评估结果
    print("\n=== 步骤6: 评估结果 ===")
    metrics = evaluate_clustering(features_full, labels_full, true_labels_full)

    # 7. 保存结果和可视化
    print("\n=== 步骤7: 保存结果 ===")

    # 生成热力图
    create_heatmap(full_data, labels_full, os.path.join(output_dir, "optics_heatmap.png"))

    # 生成可达性图（使用采样数据）
    print("生成可达性图...")
    sample_for_plot = stratified_sampling(full_data, sample_size=3000)
    features_plot = preprocess_data(sample_for_plot)
    optics_for_plot = OPTICS(**best_params, cluster_method='xi', metric='euclidean')
    optics_for_plot.fit(features_plot)
    create_reachability_plot(optics_for_plot, os.path.join(output_dir, "optics_reachability_plot.png"))

    # 保存聚类数据
    full_data_with_labels = full_data.copy()
    full_data_with_labels['cluster'] = labels_full

    # 数值列保持8位小数
    numeric_columns = full_data_with_labels.select_dtypes(include=[np.number]).columns
    for col in numeric_columns:
        if col not in ['subject_id', 'action_id', 'variant', 'cluster']:
            full_data_with_labels[col] = full_data_with_labels[col].round(8)

    full_data_with_labels.to_csv(os.path.join(output_dir, "optics_clustered_data.csv"),
                                 index=False, float_format='%.8f')
    print("OPTICS聚类数据已保存")

    # 8. 聚类分析
    print(f"\n=== OPTICS聚类分析结果 ===")
    unique_clusters = np.unique(labels_full[labels_full != -1])
    n_noise = np.sum(labels_full == -1)

    print(f"总聚类数: {len(unique_clusters)}")
    print(f"噪声点数: {n_noise} ({n_noise / len(labels_full) * 100:.2f}%)")

    # 分析每个聚类的组成
    print("\n聚类组成分析:")
    for cluster_id in sorted(unique_clusters)[:10]:
        cluster_mask = labels_full == cluster_id
        cluster_size = np.sum(cluster_mask)
        cluster_actions = full_data[cluster_mask]['action_id'].value_counts()
        dominant_action = cluster_actions.index[0]
        dominant_ratio = cluster_actions.iloc[0] / cluster_size
        print(f"聚类 {cluster_id}: 大小={cluster_size}, 主要动作={dominant_action} ({dominant_ratio:.2%})")

    # 9. 最终结果
    print(f"\n=== 最终结果 ===")
    print(f"完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(
        f"最佳参数: min_samples={best_params['min_samples']}, xi={best_params['xi']}, min_cluster_size={best_params['min_cluster_size']}")
    print(f"最终指标: silhouette={metrics['silhouette_score']:.8f}, "
          f"NMI={metrics['nmi_score']:.8f}, ARI={metrics['ari_score']:.8f}")
    print(f"结果保存在: {output_dir}")

    # 恢复输出并关闭日志
    sys.stdout = logger.terminal
    logger.close()
    print(f"运行日志已保存到: {log_file}")


if __name__ == "__main__":
    main()