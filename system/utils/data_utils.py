import os
import pandas as pd
from sklearn.model_selection import train_test_split
from collections import Counter
import numpy as np
from scipy.stats import dirichlet, beta 
import torch
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import math
import random

def plot_client_noise_ratios(clients_data, alpha, beta_param):
    """
    Simulates the noise injection process and plots pie charts for each client
    showing the ratio of Clean vs. Noisy samples.
    
    Args:
        clients_data: List of (features, labels) tuples from dirichlet_split_dataset.
        alpha: Alpha parameter for Beta distribution.
        beta_param: Beta parameter for Beta distribution.
    """
    num_clients = len(clients_data)
    
    # Generate noise proportions exactly as in beta_inject_noise_to_clients
    noise_proportions = beta.rvs(alpha, beta_param, size=num_clients)
    
    # Calculate grid size for subplots (e.g., 2 columns, or 3 columns)
    cols = 3
    rows = math.ceil(num_clients / cols)
    
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
    axes = axes.flatten()
    
    colors = ['#66b3ff', '#ff9999'] # Blue for Clean, Red for Noisy
    
    print(f"\n--- Generating Noise Distribution Plots (Alpha={alpha}, Beta={beta_param}) ---")

    for i in range(num_clients):
        features, labels = clients_data[i]
        total_samples = len(labels)
        noise_prop = noise_proportions[i]
        
        # Calculate expected number of noisy samples based on the logic in inject_noise
        num_noisy = int(noise_prop * total_samples)
        num_clean = total_samples - num_noisy
        
        # Prepare data for pie chart
        sizes = [num_clean, num_noisy]
        labels_map = ['Clean', 'Noisy']
        
        # Plot
        ax = axes[i]
        wedges, texts, autotexts = ax.pie(
            sizes, 
            labels=labels_map, 
            autopct='%1.1f%%', 
            startangle=90, 
            colors=colors,
            explode=(0, 0.1) if num_noisy > 0 else (0, 0) # Explode the noisy slice slightly
        )
        
        # Styling
        ax.set_title(f'Client {i+1}\n(Noise Rate: {noise_prop:.2%})')
        
        # Make percent text readable
        for autotext in autotexts:
            autotext.set_color('black')
            autotext.set_weight('bold')

    # Hide unused subplots if any
    for i in range(num_clients, len(axes)):
        fig.delaxes(axes[i])
        
    plt.tight_layout()
    plt.show()
def load_dataset(dataset_name):
    """
    Load a raw dataset from the 'dataset' folder.

    Args:
        dataset_name (str): The name of the dataset file without the '.pkl' extension.

    Returns:
        pd.DataFrame: The loaded DataFrame.
    """
    # Define the path to the dataset file
    dataset_path = os.path.join('dataset', rf'{dataset_name}/raw', f'{dataset_name}.pkl')
    print(dataset_path)
    # Check if the file exists
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"The dataset file '{dataset_name}.pkl' does not exist in the 'dataset' folder.")
    
    # Load the dataset
    try:
        df = pd.read_pickle(dataset_path)
    except Exception as e:
        raise Exception(f"Error loading the dataset '{dataset_name}.pkl': {e}")
    
    features, labels = df.drop('label', axis=1).values, df['label'].values
    # features, labels = df.drop('attack_cat', axis=1).values, df['attack_cat'].values

    # Return the features and labels as separate arrays

    return features, labels

def split_dataset(features, labels, test_size=0.2):
    # Split the dataset into training and test sets with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=test_size, stratify=labels
    )

    # Further split the training set into training and validation sets with stratification
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.1, stratify=y_train
    )
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def dirichlet_split_dataset(features, labels, alpha=1.0, num_clients=5):
    """
    Split the dataset using a Dirichlet distribution to simulate non-iid data.

    Args:
        features (np.ndarray): Features of the dataset.
        labels (np.ndarray): Labels of the dataset.
        alpha (float): Concentration parameter of the Dirichlet distribution.
        num_clients (int): Number of clients to split the dataset among.

    Returns:
        list: List of tuples containing features and labels for each client.
    """
    unique_labels = np.unique(labels)
    num_classes = len(unique_labels)
    label_distribution = np.zeros((num_clients, num_classes))

    # Generate a Dirichlet distribution for each client
    for i in range(num_clients):
        label_distribution[i] = dirichlet.rvs(alpha * np.ones(num_classes), size=1)[0]

    # Normalize the distribution so that the sum of probabilities for each class is 1 across all clients
    label_distribution /= label_distribution.sum(axis=0)

    # Assign samples to clients based on the generated distribution
    client_data = [([], []) for _ in range(num_clients)]
    for label in unique_labels:
        idx = np.where(labels == label)[0]
        np.random.shuffle(idx)
        proportions = label_distribution[:, unique_labels.tolist().index(label)]
        proportions = np.cumsum(proportions)
        last_idx = 0
        for client_id in range(num_clients):
            end_idx = int(proportions[client_id] * len(idx))
            client_data[client_id][0].extend(features[idx[last_idx:end_idx]])
            client_data[client_id][1].extend(labels[idx[last_idx:end_idx]])
            last_idx = end_idx

    
    return [(np.array(client_features), np.array(client_labels)) for client_features, client_labels in client_data]


def inject_noise(features, labels, noise_proportion):
    """
    Inject noise into the dataset by randomly flipping the labels.

    Args:
        features (np.ndarray): Features of the dataset.
        labels (np.ndarray): Labels of the dataset.
        noise_proportion (float): Proportion of labels to be flipped.

    Returns:
        np.ndarray: Noisy features.
        np.ndarray: Noisy labels.
    """
    noisy_labels = labels.copy()
    num_samples = len(labels)
    num_noisy_samples = int(noise_proportion * num_samples)
    
    # Randomly select samples to flip
    indices_to_flip = np.random.choice(num_samples, num_noisy_samples, replace=False)
    
    # Flip the selected labels
    unique_labels = np.unique(labels)
    for idx in indices_to_flip:
        current_label = noisy_labels[idx]
        # Select a new label different from the current one
        new_label = np.random.choice(unique_labels[unique_labels != current_label])
        noisy_labels[idx] = new_label
    
    return features, noisy_labels

def beta_inject_noise_to_clients(clients_data, alpha=2.0, beta_param=2.0):
    """
    Inject noise into each client's dataset based on a Beta distribution.

    Args:
        client_data (list): List of tuples containing features and labels for each client.
        alpha (float): Alpha parameter of the Beta distribution.
        beta_param (float): Beta parameter of the Beta distribution.

    Returns:
        list: List of tuples containing noisy features and labels for each client.
    """
    noisy_clients_data = []
    
    # Generate noise proportions for each client using the Beta distribution
    noise_proportions = beta.rvs(alpha, beta_param, size=len(clients_data))
    print(f"Noise proportions for each client: {noise_proportions}")
    for (client_features, client_labels), noise_proportion in zip(clients_data, noise_proportions):
        noisy_features, noisy_labels = inject_noise(client_features, client_labels, noise_proportion)
        noisy_clients_data.append((noisy_features, noisy_labels))
    
    return noisy_clients_data

def load_and_preprocess_data(
    dataset_name: str,
    num_clients: int,
    d_alpha: float,
    alpha: float,
    beta_param: float,
    batch_size: int = 64, 
    show_plot: bool = False
) -> tuple[list[DataLoader], DataLoader, DataLoader]:
    """
    Load, preprocess, and split the dataset into training, validation, and test sets.
    Also, inject noise into the clients' data.

    Parameters:
    - dataset_name (str): Name of the dataset to load.
    - num_clients (int): Number of clients to split the data among.
    - d_alpha (float): Dirichlet distribution alpha parameter for splitting data among clients.
    - alpha (float): Alpha parameter for noise injection.
    - beta_param (float): Beta parameter for noise injection.
    - batch_size (int): Batch size for DataLoader.

    Returns:
    - noisy_clients_loaders (list[DataLoader]): List of DataLoaders for each noisy client.
    - val_loader (DataLoader): DataLoader for validation set.
    - test_loader (DataLoader): DataLoader for test set.
    """
    try:
        features, labels = load_dataset(dataset_name)
        
        # Handle different data formats based on dataset
        if dataset_name.lower() == 'botnet':
            # For botnet dataset, keep the original feature format (tabular data)
            # No reshape needed for tabular data
            original_features = features.reshape(-1, 1, 1, 16)
        elif dataset_name.lower() == 'unsw':
            original_features = features.reshape(-1, 1, 1, 41)
        else:
            # For other datasets, apply the original reshape (assuming image-like data)
            features = features.reshape(-1, 1, 1, 49)
            original_features = features
            
        X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(original_features, labels)
        clients_data = dirichlet_split_dataset(X_train, y_train, alpha=d_alpha, num_clients=num_clients)
        # if show_plot:
        #     plot_client_noise_ratios(clients_data=clients_data, alpha=alpha, beta_param=beta_param)
        noisy_clients_data = beta_inject_noise_to_clients(clients_data=clients_data, alpha=alpha, beta_param=beta_param)
        
        # Convert the validation and test data to PyTorch DataLoader
        val_dataset = TensorDataset(torch.tensor(X_val).type(torch.float32), torch.tensor(y_val).type(torch.int64))
        test_dataset = TensorDataset(torch.tensor(X_test).type(torch.float32), torch.tensor(y_test).type(torch.int64))

        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

        # Convert noisy_clients_data to a list of DataLoaders if needed
        noisy_clients_loaders = []
        for client_id, (client_X, client_y) in enumerate(noisy_clients_data):
            client_dataset = TensorDataset(torch.tensor(client_X).type(torch.float32), torch.tensor(client_y).type(torch.int64))
            client_loader = DataLoader(client_dataset, batch_size=batch_size, shuffle=True)  # Shuffle training data
            noisy_clients_loaders.append(client_loader)

        return noisy_clients_loaders, val_loader, test_loader

    except Exception as e:
        print(f"An error occurred: {e}")
        raise





# if 
    




if __name__ == "__main__":
    # 1. 设置随机种子
    def set_random_seed(seed=0):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    set_random_seed(0)

    # 2. 生成噪声比例
    num_clients = 5
    alpha_param = 0.3
    beta_param = 0.5
    noise_proportions = beta.rvs(alpha_param, beta_param, size=num_clients)
    
    print(f"生成的噪声比例: {noise_proportions}")

    # 3. 绘图配置
    # figsize=(宽, 高): 宽度设为 3 * 客户端数量，高度固定为 4，保证单行显示不拥挤
    fig, axes = plt.subplots(1, num_clients, figsize=(3 * num_clients, 4))
    
    # 确保 axes 是可迭代的（防止只有一个客户端时报错）
    if num_clients == 1:
        axes = [axes]

    colors = ['#66b3ff', '#ff9999']  # 蓝色 (Clean), 红色 (Noisy)
    legend_labels = ['Normal', 'Noisy']
    
    # 用于存储图例句柄
    legend_wedges = []

    for i in range(num_clients):
        prop = noise_proportions[i]
        
        # 数据准备
        sizes = [1.0 - prop, prop] # [Clean, Noisy]
        
        ax = axes[i]
        
        # 炸开效果：如果有噪声，稍微分离红色部分
        explode = (0, 0.1) if prop > 0.01 else (0, 0)
        
        # 绘制饼图
        # labels=None: 不在饼图周围显示 Clean/Noisy 文字，依靠图例
        wedges, texts, autotexts = ax.pie(
            sizes, 
            labels=None, 
            autopct='%1.1f%%', 
            startangle=90, 
            colors=colors,
            explode=explode,
            shadow=True,
            textprops={'fontsize': 10, 'weight': 'bold', 'color': 'white'} # 百分比字体白色加粗
        )
        
        # 修正百分比文字颜色：如果是浅色背景，字变黑；深色背景字变白
        # 这里统一把文字设为黑色以便阅读
        for autotext in autotexts:
            autotext.set_color('black')

        # 设置每个子图的标题
        ax.set_title(f'Client {i+1}', fontsize=11)
        
        # 保存第一个图的 wedge 用于生成全局图例
        if i == 0:
            legend_wedges = wedges

    # 4. 添加全局图例 (Single Row Legend)
    # bbox_to_anchor 控制图例位置，ncol=2 让图例横向排列
    fig.legend(legend_wedges, legend_labels, loc='upper center', 
               bbox_to_anchor=(0.5, 1.05), ncol=2, fontsize=12, frameon=False)

    # 调整布局，给图例留出顶部空间
    plt.tight_layout(rect=[0, 0.05, 1, 0.9]) 

    # 5. 保存图片
    save_dir = "plots"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    filename = f"client_noise_row_beta_{alpha_param}_{beta_param}.pdf"
    save_path = os.path.join(save_dir, filename)
    
    # bbox_inches='tight' 会自动裁剪周围空白，包含外部的图例
    plt.savefig(save_path, dpi=300, bbox_inches='tight',transparent=True)
    
    print(f"图片已保存至: {save_path}")
    plt.show()

#     dataset = 'cicids2017'
#     # features, labels = load_dataset(dataset)
#     # print(f"Features shape: {features.shape}")
#     # print(f"Labels shape: {labels.shape}")

#     # X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(features, labels)

#     # print(f"Training features shape: {X_train.shape}")
#     # print(f"Validation features shape: {X_val.shape}")
#     # print(f"Testing features shape: {X_test.shape}")
#     # # Dirichlet split for training set
#     # dirichlet_splits = dirichlet_split_dataset(X_train, y_train, alpha=0.1, num_clients=5)
#     # for i, (client_features, client_labels) in enumerate(dirichlet_splits):
#     #     # print(f"Dirichlet Client {i+1} training features shape: {client_features.shape}")
#     #     # print(f"Dirichlet Client {i+1} training labels shape: {client_labels.shape}")
#     #     print(f"Dirichlet Client {i+1} training label distribution: {Counter(client_labels)}")

#     # noisy_clients_data = beta_inject_noise_to_clients(dirichlet_splits,alpha=0.1, beta_param=0.1)
#     load_and_preprocess_data(dataset_name=dataset,num_clients=5,d_alpha=0.7,alpha=0.1, beta_param=0.1)











