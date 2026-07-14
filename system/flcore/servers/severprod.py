import time
import os
import json
from flcore.clients.clientprod import clientFedProD
import copy
from flcore.servers.serverbase import Server
from threading import Thread
import random
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader
from torch import stack, no_grad
from torch.optim import SGD, Adam, lr_scheduler
from torch.nn.functional import softmax, log_softmax
import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm


class FedProD(Server):
    def __init__(self, args, times):
        super().__init__(args, times)
        # self.global_student = self.global_model
        self.threshold = self.args.threshold
        self.lamda = args.lamda
        # self.global_teacher_good = copy.deepcopy(self.global_model)
        # self.global_teacher_bad = copy.deepcopy(self.global_model)
        self.optimizer = Adam(self.global_model.parameters(), lr=self.args.global_learning_rate, weight_decay=0.002)
        
        # 设置日志目录
        log_dir = f"./logs/{args.dataset}/{self.algorithm}_hyper"
        os.makedirs(log_dir, exist_ok=True)
        
        # 初始化日志记录 - 使用参数命名日志文件
        log_filename = f"training_log_nc{args.num_clients}_M{args.M}_d{args.d_alpha}_alpha{args.alpha_param}_beta{args.beta_param}_lamda{args.lamda}_th{args.threshold}.json"
        self.log_file = os.path.join(log_dir, log_filename)
        self.training_log = {
            "rounds": [],
            "test_acc": [],
            "test_pre": [],
            "test_recall": [],
            "test_f1_score": [],
            "test_loss": [],
            "train_loss": [],
            "n_kl_loss": [],
            "p_kl_loss": []
        }
        
        # select slow clients
        self.set_slow_clients()
        self.set_clients(clientFedProD)
        # loss
        self.loss = torch.nn.CrossEntropyLoss()
        # distill
        self.T = self.args.temp
        # self.mini_batch_size_distill = self.args.mini_batch_size_distill
        self.warmup_rounds = self.args.warm_up
        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")
        
        # self.load_model()
        self.Budget = []
        
    def log_metrics(self, round_num, test_acc, test_pre, test_recall, test_f1_score, test_loss, train_loss, n_kl_loss=0.0, p_kl_loss=0.0):
        """记录训练指标到日志文件"""
        self.training_log["rounds"].append(round_num)
        self.training_log["test_acc"].append(test_acc)
        self.training_log["test_pre"].append(test_pre)
        self.training_log["test_recall"].append(test_recall)
        self.training_log["test_f1_score"].append(test_f1_score)
        self.training_log["test_loss"].append(test_loss)
        self.training_log["train_loss"].append(train_loss)
        self.training_log["n_kl_loss"].append(n_kl_loss)
        self.training_log["p_kl_loss"].append(p_kl_loss)
        
        # 保存到文件
        with open(self.log_file, 'w') as f:
            json.dump(self.training_log, f, indent=2)
            
    def aggregate_warm_parameters(self):
        super().aggregate_parameters()
        # self.global_student = copy.deepcopy(self.global_model)

    def get_bad_logit(self, batch, bad_id):
        list_softmax = []
        for id in bad_id:
            bad_teacher = self.clients[id].model
            bad_teacher.to(self.device)
            bad_teacher.eval()
            with torch.no_grad():
                
                _, local_logits = bad_teacher(batch)
                local_logits = torch.reciprocal(local_logits)
                local_softmax = softmax(local_logits,dim=-1)
                list_softmax.append(local_softmax/self.T)

        return list_softmax

    def compute_uncertainty(self):
        self.uncertainty_dict = {}
        for client in self.clients:
            self.uncertainty_dict[client.id] = (client.compute_uncertainty(self.val_dataloader))
        bad, good = [], []
        for key in self.uncertainty_dict.keys():
            print(f"Client {key} has uncertainty {self.uncertainty_dict[key]}")
        for id, un in self.uncertainty_dict.items():
            if un > self.threshold:
                bad.append(id)
            else:
                good.append(id)
        return bad, good

        
    def aggregate_good_clients(self, good_clients, good_clients_weights):
        for param in self.global_model.parameters():
            param.data.zero_()
        
        for w, client_model in zip(good_clients_weights, good_clients):
            print(f'train sample weights {w}')
            self.add_good_parameters(w, client_model)
    def add_good_parameters(self, w, client_model):
        for server_param, client_param in zip(self.global_model.parameters(), client_model.parameters()):
            server_param.data += w * client_param.data.clone()

    def distillation_with_bad(self, bad_ids):
        self.global_model.to(self.device)
        
        for c_id in bad_ids:
            client = self.clients[c_id]
            client.model.eval()
            client.model.to(self.device)
            for x, _ in tqdm(self.val_dataloader, desc='distillation with bad client {id}'):
                x = x.to(self.device)
                _, bad_protos = client.model.base(x)
                _, protos = self.global_model.base(x)
                bad_protos = bad_protos.detach()
                bad_protos = F.normalize(bad_protos, p=2, dim=1).to(self.device)
                protos = F.normalize(protos, p=2, dim=1).to(self.device)
                mse_loss = -F.mse_loss(bad_protos, protos, reduction='mean')
                self.optimizer.zero_grad()
                mse_loss.backward()
                self.optimizer.step()

           
            

    def aggregate_distillation(self):
        # self.aggregate_good_clients()
        for step in range(10):
            for x, _ in tqdm(self.val_dataloader, desc=f"Distillation Step {step} "):
                x = x.to(self.device)
                all_bad_probs = self.get_bad_logit(x, self.bad_ids)
                _, logits = self.global_model(x)
                good_probs = torch.softmax(logits /self.T, dim=1)
                loss = 0.
                if len(all_bad_probs) > 0:
                    for bad_probs in all_bad_probs:
                        bad_probs = bad_probs.to(self.device)
                        bad_teacher_loss = F.kl_div(good_probs.log(), bad_probs, reduction='batchmean')
                        loss += bad_teacher_loss
                    loss = loss / len(all_bad_probs)  
                else:
                    bad_probs = torch.zeros((128, 15)).to(self.device)
                    loss = F.kl_div(good_probs.log(), bad_probs, reduction='batchmean')
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
    
    def extract_features_for_tsne(self, dataloader, max_samples=1000):
        # """提取中间层特征用于t-SNE可视化"""
        self.global_model.eval()
        self.global_model.to(self.device)
        
        all_features = []
        all_labels = []
        all_client_ids = []
        
        collected = 0
        with torch.no_grad():
            for x, y in dataloader:
                if collected >= max_samples:
                    break
                x = x.to(self.device)
                y = y.to(self.device)
                
                _, features = self.global_model.base(x)
                all_features.append(features.cpu())
                all_labels.append(y.cpu())
                
                batch_size = x.shape[0]
                all_client_ids.extend([-1] * batch_size)  # -1 表示全局模型
                collected += batch_size
        
        features = torch.cat(all_features, dim=0).numpy()
        labels = torch.cat(all_labels, dim=0).numpy()
        
        return features, labels, np.array(all_client_ids)


    def extract_client_features_for_tsne(self, max_samples=200):
        """提取所有客户端的特征用于t-SNE"""
        all_features = []
        all_labels = []
        all_client_ids = []
        
        for client in self.clients[:5]:  # 取前5个客户端展示
            client.model.eval()
            client.model.to(self.device)
            
            collected = 0
            trainloader = self.client_train_dataloaders[client.id]
            with torch.no_grad():
                for x, y in trainloader:
                    if collected >= max_samples // 5:
                        break
                    x = x.to(self.device)
                    y = y.to(self.device)
                    
                    _, features = client.model.base(x)
                    all_features.append(features.cpu())
                    all_labels.append(y.cpu())
                    all_client_ids.extend([client.id] * x.shape[0])
                    collected += x.shape[0]
        
        return (torch.cat(all_features, dim=0).numpy(),
                torch.cat(all_labels, dim=0).numpy(),
                np.array(all_client_ids))


    def save_tsne_data(self, round_num, features, labels, client_ids, save_dir="tsne_data"):
        """保存t-SNE数据到文件"""
        os.makedirs(save_dir, exist_ok=True)
        
        import pickle
        data = {
            'round': round_num,
            'features': features,
            'labels': labels,
            'client_ids': client_ids
        }
        
        filename = f"tsne_round_{round_num}.pkl"
        filepath = os.path.join(save_dir, filename)
        
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        
        print(f"t-SNE data saved to {filepath}")


    def train(self):
        self.bad_ids = set()
        timing_log = {
            "rounds": [], "client_train_time": [], "mc_dropout_time": [],
            "server_aggregation_time": [], "total_round_time": []
        }
        for i in range(self.global_rounds+1):
            round_start = time.time()
            self.selected_clients = self.select_clients()
            self.send_models()
            test_acc_list = []
            test_pre_list = []
            test_recall_list = []
            test_f1_score_list = []
            test_loss = 0.0
            train_loss = 0.0
            client_train_time = 0.0
            mc_dropout_time = 0.0
            server_agg_time = 0.0

            # t-SNE
            # 在 train() 方法中最后几个round保存t-SNE数据
            if i % 50 == 0 or i == self.global_rounds:  # 每50轮或最后一轮保存
                features, labels, client_ids = self.extract_features_for_tsne(self.test_dataloader)
                self.save_tsne_data(i, features, labels, client_ids)
                
                # 客户端特征
                c_features, c_labels, c_ids = self.extract_client_features_for_tsne()
                self.save_tsne_data(f"client_{i}", c_features, c_labels, c_ids)


            if i%self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate models")
                test_acc, test_pre, test_recall, test_f1_score, test_loss = self.evaluate()
                test_acc_list.append(test_acc)
                test_pre_list.append(test_pre)
                test_recall_list.append(test_recall)
                test_f1_score_list.append(test_f1_score)
            
            if i <= self.warmup_rounds:
                train_loss = 0.0
                p_kl_loss = 0.0
                n_kl_loss = 0.0
                print(f"\n-------------Warm up round number: {i}-------------")
                
                train_start = time.time()
                for client in self.selected_clients:
                    l1, l2, l3  = client.train(self.client_train_dataloaders[client.id])
                    train_loss += l1
                    n_kl_loss += l2
                    p_kl_loss += l3
                client_train_time = time.time() - train_start
                mc_dropout_time = 0.0
                
                avg_train_loss = train_loss/len(self.selected_clients)
                avg_n_kl_loss = n_kl_loss/len(self.selected_clients)
                avg_p_kl_loss = p_kl_loss/len(self.selected_clients)
                print(f"Average Train Loss: {avg_train_loss}")
                
                # 记录指标到日志文件
                if i%self.eval_gap == 0:
                    self.log_metrics(i, test_acc, test_pre, test_recall, test_f1_score, test_loss, avg_train_loss, avg_n_kl_loss, avg_p_kl_loss)
                
                agg_start = time.time()
                self.receive_models()
                self.aggregate_parameters()
                server_agg_time = time.time() - agg_start
            else:
                print(f'en clients: {self.bad_ids}')
                
                train_loss = 0.0
                p_kl_loss = 0.0
                n_kl_loss = 0.0
                
                train_start = time.time()
                for client in self.selected_clients:
                    l1, l2, l3  = client.train(self.client_train_dataloaders[client.id])
                    train_loss += l1
                    n_kl_loss += l2
                    p_kl_loss += l3
                client_train_time = time.time() - train_start
                
                avg_train_loss = train_loss/len(self.selected_clients)
                avg_n_kl_loss = n_kl_loss/len(self.selected_clients)
                avg_p_kl_loss = p_kl_loss/len(self.selected_clients)
                print(f"Average Train Loss: {avg_train_loss}")
                
                # 记录指标到日志文件
                if i%self.eval_gap == 0:
                    self.log_metrics(i, test_acc, test_pre, test_recall, test_f1_score, test_loss, avg_train_loss, avg_n_kl_loss, avg_p_kl_loss)
                
                mc_start = time.time()
                bad, good = self.compute_uncertainty()
                mc_dropout_time = time.time() - mc_start
                
                print(f'bad clients: {bad}')
                print(f'good clients: {good}')
                self.bad_ids = set(bad)
                
                agg_start = time.time()
                self.receive_models()
                good_clients = []
                good_clients_weights = []
                for id in good:
                    good_clients.append(self.clients[id].model)
                    good_clients_weights.append(self.clients[id].train_samples)
                good_clients_weights = [w/sum(good_clients_weights) for w in good_clients_weights]
                self.aggregate_good_clients(good_clients, good_clients_weights)
                # self.distillation_with_bad(self.bad_ids)
                # self.aggregate_distillation()
                self.collect_protos()
                self.aggregate_good_protos(good_ids=good)
                self.aggregate_bad_protos(bad_ids=bad)
                self.send_protos()
                server_agg_time = time.time() - agg_start

            total_round_time = time.time() - round_start
            timing_log["rounds"].append(i)
            timing_log["client_train_time"].append(client_train_time)
            timing_log["mc_dropout_time"].append(mc_dropout_time)
            timing_log["server_aggregation_time"].append(server_agg_time)
            timing_log["total_round_time"].append(total_round_time)

            if i % self.eval_gap == 0:
                print(f"\n--- Round {i} Timing ---")
                print(f"  Client Local Training:        {client_train_time:.4f}s")
                print(f"  Client Inference (MC Dropout): {mc_dropout_time:.4f}s")
                print(f"  Server Aggregation:           {server_agg_time:.4f}s")
                print(f"  Total Round Time:             {total_round_time:.4f}s")
                self.save_detailed_metrics(i)

        print('final test acc: ', sum(test_acc_list[-5:]) / 5)
        print('final test pre: ', sum(test_pre_list[-5:]) / 5)
        print('final test recall: ', sum(test_recall_list[-5:]) / 5)
        print('final test f1: ', sum(test_f1_score_list[-5:]) / 5)

        self.save_detailed_metrics(i)

        timing_file = self.log_file.replace('.json', '_timing.json')
        with open(timing_file, 'w') as f:
            json.dump(timing_log, f, indent=2)
        print(f"\nTiming log saved to {timing_file}")

        print("\n========== Average Timing Summary (all rounds) ==========")
        print(f"  Client Local Training:        {sum(timing_log['client_train_time'])/len(timing_log['client_train_time']):.4f}s")
        print(f"  Client Inference (MC Dropout): {sum(timing_log['mc_dropout_time'])/len(timing_log['mc_dropout_time']):.4f}s")
        print(f"  Server Aggregation:           {sum(timing_log['server_aggregation_time'])/len(timing_log['server_aggregation_time']):.4f}s")
        print(f"  Total Round Time:             {sum(timing_log['total_round_time'])/len(timing_log['total_round_time']):.4f}s")
        print("=========================================================")

    # def aggreate_protos(bad_ids, good_ids):
    #     pass

    def compute_good_proto_weights(self,good_ids, uncertainty_dict):
        print('compute good proto weights...')
        # 初始化输出权重字典
        weights = {}
        
        if not good_ids:
            return weights
        
        # 第一步：归一化不确定性（映射到[0,1]区间）
        all_uncertainties = np.array([uncertainty_dict[client_id] for client_id in uncertainty_dict])
        min_unc, max_unc = np.min(all_uncertainties), np.max(all_uncertainties)
        
        # 线性归一化公式：(x - min) / (max - min + eps)
        normalized_unc = {}
        eps = 1e-10  # 防止除零
        for client_id in uncertainty_dict:
            if max_unc > min_unc:
                normalized_unc[client_id] = (uncertainty_dict[client_id] - min_unc) / (max_unc - min_unc + eps)
            else:  # 所有客户端uncertainty相同的情况
                normalized_unc[client_id] = 0.5  # 赋予中性值
        
        # 第二步：仅对good_ids计算权重
        total_weight = 0.0
        raw_weights = {}
        
        for client_id in good_ids:
            
            # 权重计算公式：数据量 * (1 - 归一化不确定性)
            raw_weight = self.clients[client_id].train_samples * (1.0 - normalized_unc[client_id])
            raw_weights[client_id] = raw_weight
            total_weight += raw_weight
        
        # 第三步：归一化权重（总和=1）
        if total_weight > 0:
            for client_id in raw_weights:
                weights[client_id] = raw_weights[client_id] / total_weight
        print(f'good weights: {weights}')
        return weights
    
    def compute_bad_proto_weights(self, bad_ids, uncertainty_dict):
        print('compute bad proto weights...')
        if not bad_ids:
            return {}
        
        # 1. 提取坏客户端的不确定性值
        bad_uncertainties = {k: uncertainty_dict[k] for k in bad_ids}
        
        # 2. 直接使用原始不确定性（或归一化到[0,1]区间）
        uncertainties = np.array(list(bad_uncertainties.values()))
        
        # 3. 权重 = 不确定性（若需归一化则启用以下代码）
        if np.max(uncertainties) > 0:
            weights = {client_id: unc / np.sum(uncertainties) for client_id, unc in bad_uncertainties.items()}
        else:  # 所有不确定性为0时均分权重
            weights = {client_id: 1.0 / len(bad_ids) for client_id in bad_ids}
        print(f'bad weights: {weights}')
        return weights
    
    def aggregate_good_protos(self, good_ids):
        """
        Aggregate the prototypes of "good" clients for each label.

        Args:
            good_ids (list): List of client IDs considered as "good".

        Returns:
            dict: A dictionary where keys are labels and values are the aggregated prototypes.
        """
        # Initialize a dictionary to store the aggregated prototypes
        self.global_good_protos = {}
        print('aggregate good protos...')
        # Compute the weights for each good client
        weights = self.compute_good_proto_weights(good_ids, self.uncertainty_dict)

        # Collect all unique labels from the "good" clients
        all_labels = set()
        for client_id in good_ids:
            all_labels.update(self.clients[client_id].protos.keys())

        # Aggregate prototypes for each label
        for label in all_labels:
            weighted_proto_sum = 0
            weight_sum = 0

            for client_id in good_ids:
                client_protos = self.clients[client_id].protos
                if label in client_protos:
                    # Add the weighted contribution of this client's prototype
                    weighted_proto_sum += weights[client_id] * client_protos[label]
                    weight_sum += weights[client_id]

            # Compute the weighted average for this label
            if weight_sum > 0:
                self.global_good_protos[label] = weighted_proto_sum / weight_sum

    def aggregate_bad_protos(self, bad_ids):
        """
        Aggregate the prototypes of "bad" clients for each label.

        Args:
            bad_ids (list): List of client IDs considered as "bad".

        Returns:
            dict: A dictionary where keys are labels and values are the aggregated prototypes.
        """
        # Initialize a dictionary to store the aggregated prototypes
        self.global_bad_protos = {}
        print('aggregate bad protos...')
        # Compute the weights for each bad client (if needed)
        # Here, we assume equal weight for simplicity, but you can customize this.
        weights = self.compute_bad_proto_weights(bad_ids, self.uncertainty_dict) 

        # Collect all unique labels from the "bad" clients
        all_labels = set()
        for client_id in bad_ids:
            all_labels.update(self.clients[client_id].protos.keys())

        # Aggregate prototypes for each label
        for label in all_labels:
            weighted_proto_sum = 0
            weight_sum = 0

            for client_id in bad_ids:
                client_protos = self.clients[client_id].protos
                if label in client_protos:
                    # Add the weighted contribution of this client's prototype
                    weighted_proto_sum += weights[client_id] * client_protos[label]
                    weight_sum += weights[client_id]

            # Compute the weighted average for this label
            if weight_sum > 0:
                self.global_bad_protos[label] = weighted_proto_sum / weight_sum
   
    def collect_protos(self):
        for client in self.clients:
            client.collect_protos(self.client_train_dataloaders[client.id])

    def send_protos(self):
        print("Sending global prototypes to clients...")
        for client in self.clients:
            client.global_good_protos = self.global_good_protos
            client.global_bad_protos = self.global_bad_protos
