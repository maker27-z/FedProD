import numpy as np
import time
from flcore.clients.clientbase import Client
from tqdm import tqdm
import copy
import torch
from torch import nn
from collections import defaultdict
from torch.nn import functional as F
# import wandb

# from torch.utils.data import DataLoader


class clientFedProD(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)

        self.lamda = args.lamda
        self.args = args
        self._global_good_protos = None
        self._global_bad_protos = None
        
        # 缓存优化：预先计算一些常用值
        self.cached_good_labels = set()
        self.cached_bad_labels = set()
        self.proto_cache_valid = False
        
    @property
    def global_good_protos(self):
        return self._global_good_protos
    
    @global_good_protos.setter
    def global_good_protos(self, value):
        self._global_good_protos = value
        self.proto_cache_valid = False  # 使缓存失效
    
    @property
    def global_bad_protos(self):
        return self._global_bad_protos
    
    @global_bad_protos.setter
    def global_bad_protos(self, value):
        self._global_bad_protos = value
        self.proto_cache_valid = False  # 使缓存失效
        
    def update_proto_cache(self):
        """更新prototype标签缓存以加速查找"""
        if self.global_good_protos is not None:
            self.cached_good_labels = set(self.global_good_protos.keys())
        else:
            self.cached_good_labels = set()
            
        if self.global_bad_protos is not None:
            self.cached_bad_labels = set(self.global_bad_protos.keys())
        else:
            self.cached_bad_labels = set()
        
        self.proto_cache_valid = True
    def train(self, trainloader):
        # trainloader = self.load_train_data()
        # self.model.to(self.device)
        self.model.train()
        
        # 更新prototype缓存以加速查找
        if not self.proto_cache_valid:
            self.update_proto_cache()

        # if self.train_slow:
        #     max_local_epochs = np.random.randint(1, max_local_epochs // 2)

        max_local_epochs = 1

        for epoch in range(max_local_epochs):
            epoch_loss = 0.0  # Initialize the total loss for the epoch
            epoch_loss_ce = 0.0  # Initialize the total cross-entropy loss for the epoch
            epoch_loss_p_kl = 0.0  # Initialize the total positive KL divergence loss for the epoch
            epoch_loss_n_kl = 0.0  # Initialize the total negative KL divergence loss for the epoch
            batch_count_p_kl = 0
            batch_count_n_kl = 0
            batch_count = 0   # Record the number of processed batches
            ce_loss = None
            p_kl_loss = None
            n_kl_loss = None
            
            for i, (x, y) in enumerate(tqdm(trainloader, desc=f"client {self.id} Training:")):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                # if self.train_slow:
                #     time.sleep(0.1 * np.abs(np.random.rand()))
                _, rep = self.model.base(x)
                output = self.model.head(rep)  
                ce_loss = self.loss(output, y)
                
                # #tSNE
                # all_features.append(rep.cpu())
                # all_labels.append(y)

                # all_features = torch.cat(all_features)
                # all_labels = torch.cat(all_labels)

                # torch.save({
                #     "features": all_features,
                #     "labels": all_labels
                # }, "FedProD_features.pt")

                # 合并计算KL散度以提高效率
                p_kl_loss = None
                n_kl_loss = None
                
                # 概率性计算：随机选择计算哪个KL散度（可选优化）
                compute_both = True  # 设为False启用概率性计算
                if not compute_both:
                    choice = torch.rand(1).item()
                    compute_good = choice > 0.5
                else:
                    compute_good = True
                
                # 使用缓存加速索引查找
                good_indices = []
                bad_indices = []
                
                # 计算positive KL散度
                if self.global_good_protos is not None and compute_good and len(self.cached_good_labels) > 0:
                    # 使用缓存的标签集合加速查找
                    good_indices = [i for i, label in enumerate(y) if int(label.item()) in self.cached_good_labels]
                
                # 计算negative KL散度
                if self.global_bad_protos is not None and (compute_both or not compute_good) and len(self.cached_bad_labels) > 0:
                    # 使用缓存的标签集合加速查找
                    bad_indices = [i for i, label in enumerate(y) if int(label.item()) in self.cached_bad_labels]
                
                # 计算positive KL散度
                if self.args.isPositive == 1:
                    if len(good_indices) > 0:
                        if i == 0:
                            print(f'client {self.id} is pos-dis...')
                        valid_local = rep[good_indices]
                        valid_global = torch.stack([self.global_good_protos[int(y[i].item())] for i in good_indices])
                        
                        local_log_probs = F.log_softmax(valid_local, dim=1)
                        global_probs = F.softmax(valid_global, dim=1)
                        
                        p_kl_loss = F.kl_div(local_log_probs, global_probs, reduction='batchmean')
                        epoch_loss_p_kl += p_kl_loss.detach().item()
                        batch_count_p_kl += 1
                    else:
                        if i == 0:
                            print(f'!!!!!!warning!!!!!!: client {self.id} is no pos-dis...')
                else:
                    print(f"client {self.id} is no pos-dis...")
                    pass
                # 计算negative KL散度
                if self.args.isNegative == 1:
                    if len(bad_indices) > 0:
                        if i == 0:
                            print(f'client {self.id} is neg-dis...')
                        valid_local = rep[bad_indices]
                        bad_protos = torch.stack([self.global_bad_protos[int(y[i].item())] for i in bad_indices])
                        bad_protos_clamped = torch.clamp(bad_protos, min=1e-8)
                        valid_global = torch.reciprocal(bad_protos_clamped)
                        
                        local_log_probs = F.log_softmax(valid_local, dim=1)
                        global_probs = F.softmax(valid_global, dim=1)
                        
                        n_kl_loss = F.kl_div(local_log_probs, global_probs, reduction='batchmean')
                        epoch_loss_n_kl += n_kl_loss.detach().item()
                        batch_count_n_kl += 1
                    else:
                        if i == 0:
                            print(f'!!!!!!warning!!!!!!: client {self.id} is no neg-dis...')
                else:
                    print(f"client {self.id} is no neg-dis...")
                    pass
                # 修复损失函数逻辑错误
                if p_kl_loss is None and n_kl_loss is None:
                    loss = ce_loss
                elif p_kl_loss is not None and n_kl_loss is None:
                    loss = ce_loss * 0.75 + p_kl_loss * self.lamda
                elif p_kl_loss is None and n_kl_loss is not None:
                    loss = ce_loss * 0.75 + n_kl_loss * self.lamda
                else:  # 同时存在p_kl_loss和n_kl_loss
                    loss = ce_loss * 0.5 + p_kl_loss * self.lamda + n_kl_loss * self.lamda
                epoch_loss += loss.detach().item()  # Accumulate the loss
                batch_count += 1           # Increment the batch count
                self.optimizer.zero_grad()  # Zero the gradients
                loss.backward()            # Backpropagate the loss
                
                # 添加梯度裁剪防止梯度爆炸
                # torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                self.optimizer.step()      # Update the weights

            avg_epoch_loss = epoch_loss / batch_count if batch_count > 0 else 0  # Calculate the average loss
            # print(f"Epoch {epoch + 1}, Average Loss: {avg_epoch_loss}")
            avg_epoch_loss_n_kl = epoch_loss_n_kl / batch_count_n_kl if batch_count_n_kl > 0 else 0  # Calculate the average loss
            # print(f"Epoch {epoch + 1}, Average KL Loss: {avg_epoch_loss_n_kl}")
            avg_epoch_loss_p_kl = epoch_loss_p_kl / batch_count_p_kl if batch_count_p_kl > 0 else 0  # Calculate the average loss

        # self.model.cpu()

        if self.learning_rate_decay:
            self.learning_rate_scheduler.step()

        # self.train_time_cost['num_rounds'] += 1
        # self.train_time_cost['total_cost'] += time.time() - start_time
        return avg_epoch_loss, avg_epoch_loss_n_kl, avg_epoch_loss_p_kl

    
    def bad_train(self, trainloader):

        self.model.train()

        if self.train_slow:
            max_local_epochs = np.random.randint(1, max_local_epochs // 2)

        for epoch in range(max_local_epochs):
            for i, (x, y) in enumerate(tqdm(trainloader, desc=f"Epoch {epoch} ")):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)

                _, logits = self.model(x)
                pseudo_label = torch.softmax(logits.detach(), self.T, dim=-1)
                max_probs, targets = torch.max(pseudo_label, dim=-1)
                mask = max_probs.ge(self.threshold).float()
                loss = (self.loss(logits, targets) * mask).mean()
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
    
    def compute_uncertainty(self, disstill_set):
        self.model.train()
        total_entropy = 0.0
        total_samples = 0
        mc_samples = self.args.M
        with torch.no_grad():
            for x, _ in tqdm(disstill_set, desc=f'Client {self.id} Computing Uncertainty'):  # 正常批次遍历
                x = x.to(self.device)
                batch_size = x.shape[0]
                # num_classes = self.model(x[:1]).shape[1]  # 动态获取类别数
                
                # 预分配结果张量 [mc_samples, batch_size, num_classes]
                mc_probs = torch.zeros((mc_samples, batch_size, self.args.num_classes), 
                                    device=self.device)
                
                # 循环MC采样
                for i in range(mc_samples):#for i in range(10):
                    y_drop, _ = self.model.base(x)
                    y_drop = self.model.head(y_drop)  # 带Dropout的前向
                    mc_probs[i] = torch.softmax(y_drop, dim=1)
                    
                
                # 计算当前批次的熵 [batch_size]
                mean_probs = mc_probs.mean(dim=0)
                entropy = -torch.sum(mean_probs * torch.log(mean_probs + 1e-10), dim=1)
                
                total_entropy += entropy.sum().item()
                total_samples += batch_size
                
        return total_entropy / max(total_samples, 1)  # 防零除

    def collect_protos(self,trainloader):
        # trainloader = self.load_train_data()
        self.model.eval()

        protos = defaultdict(list)
        with torch.no_grad():
            for i, (x, y) in enumerate(tqdm(trainloader, desc=f'Client {self.id} computing protos')):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))
                _, rep = self.model.base(x)

                for i, yy in enumerate(y):
                    y_c = yy.item()
                    protos[y_c].append(rep[i, :].detach().data)

        self.protos = agg_func(protos)

                
                



def agg_func(protos):
    """
    Returns the average of the weights.
    """

    for [label, proto_list] in protos.items():
        if len(proto_list) > 1:
            proto = 0 * proto_list[0].data
            for i in proto_list:
                proto += i.data
            protos[label] = proto / len(proto_list)
        else:
            protos[label] = proto_list[0]

    return protos
