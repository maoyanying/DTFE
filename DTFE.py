"""
here is the mian backbone for DTFE
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from ...subNets import BertTextEncoder
from ...subNets.transformers_encoder.transformer import TransformerEncoder   

class ExternalMemory(nn.Module):
    def __init__(self, text_dim, mem_size):
        super(ExternalMemory, self).__init__()
        self.text_dim = text_dim
        self.mem_size = mem_size
        
        # 外部记忆矩阵
        self.memory = nn.Parameter(torch.randn(mem_size, text_dim))
        
        # Linear transformations
        self.query_transform = nn.Linear(text_dim, text_dim)
        #self.key_transform = nn.Linear(text_dim, text_dim)
        #self.value_transform = nn.Linear(text_dim, text_dim)
        self.key_transform = nn.Linear(mem_size, text_dim)
        self.value_transform = nn.Linear(mem_size, text_dim)
        
    def forward(self, query):
        # 将输入的查询向量query进行线性变换，得到变换后的查询向量。这个变换的目的是将查询向量调整到与记忆矩阵相同的维度，以便进行后续的注意力计算。
        query = self.query_transform(query)  # (batch_size, text_dim)
        # 使用self.key_transform(self.memory)和self.value_transform(self.memory)分别对外部记忆矩阵进行键和值的线性变换，
        # 得到变换后的键矩阵和值矩阵。这些变换将外部记忆中的每个条目映射到与查询向量相同的空间。
        key = self.key_transform(self.memory)  # (mem_size, text_dim)
        value = self.value_transform(self.memory)  # (mem_size, text_dim)
        
        # 通过矩阵乘法torch.matmul(query, key.t())计算查询向量与每个键之间的点积注意力分数。
        # 这里使用了缩放点积注意力机制，通过除以math.sqrt(self.text_dim)来缩放注意力分数，其中self.text_dim是查询和键的维度大小
        scores = torch.matmul(query, key.t()) / math.sqrt(self.text_dim)  # (batch_size, mem_size)

        # 对注意力分数应用softmax函数F.softmax(scores, dim=-1)，得到注意力权重，表示每个外部记忆条目对查询的重要性。
        attention_weights = F.softmax(scores, dim=-1)  # (batch_size, mem_size)
        
        # 使用注意力权重对值矩阵进行加权求和，计算加权平均值。weighted_sum表示基于注意力机制加权后的记忆信息.
        weighted_sum = torch.matmul(attention_weights.unsqueeze(1), value.unsqueeze(0))  # (batch_size, 1, text_dim)
        weighted_sum = weighted_sum.squeeze(1)  # (batch_size, text_dim)
        
        return weighted_sum

class DTFE(nn.Module):
    def __init__(self, args, ablation_mode=None):
        super(DTFE, self).__init__()
        if args.use_bert:
            self.text_model = BertTextEncoder(use_finetune=args.use_finetune, transformers=args.transformers,
                                              pretrained=args.pretrained)
        self.use_bert = args.use_bert
        dst_feature_dims, nheads = args.dst_feature_dim_nheads
        if args.dataset_name == 'mosi':
            if args.need_data_aligned:
                self.len_l, self.len_v, self.len_a = 50, 50, 50
            else:
                self.len_l, self.len_v, self.len_a = 50, 500, 375
        if args.dataset_name == 'mosei':
            if args.need_data_aligned:
                self.len_l, self.len_v, self.len_a = 50, 50, 50
            else:
                self.len_l, self.len_v, self.len_a = 50, 500, 500
        self.ablation_mode = ablation_mode # MYY例如: None, 'text_only', 'no_audio', 'text_visual'
        self.orig_d_l, self.orig_d_a, self.orig_d_v = args.feature_dims
        self.d_l = self.d_a = self.d_v = dst_feature_dims
        self.num_heads = nheads     
        self.layers = args.nlevels 
        self.attn_dropout = args.attn_dropout
        self.attn_dropout_a = args.attn_dropout_a
        self.attn_dropout_v = args.attn_dropout_v
        self.relu_dropout = args.relu_dropout
        self.embed_dropout = args.embed_dropout
        self.res_dropout = args.res_dropout
        self.output_dropout = args.output_dropout
        self.text_dropout = args.text_dropout
        self.attn_mask = args.attn_mask
        combined_dim_low = self.d_a    
        combined_dim_high = self.d_a 
        combined_dim = (self.d_l + self.d_a + self.d_v ) + self.d_l * 3  
        
        output_dim = 1

        # 1. Temporal convolutional layers for initial feature
        self.proj_l = nn.Conv1d(self.orig_d_l, self.d_l, kernel_size=args.conv1d_kernel_size_l, padding=0, bias=False)
        self.proj_a = nn.Conv1d(self.orig_d_a, self.d_a, kernel_size=args.conv1d_kernel_size_a, padding=0, bias=False)
        self.proj_v = nn.Conv1d(self.orig_d_v, self.d_v, kernel_size=args.conv1d_kernel_size_v, padding=0, bias=False)

        # 2. Modality-specific encoder
        self.encoder_s_l = self.get_network(self_type='l', layers = self.layers)       
        self.encoder_s_v = self.get_network(self_type='v', layers = self.layers)
        self.encoder_s_a = self.get_network(self_type='a', layers = self.layers)

        #   Modality-shared encoder 
        self.encoder_c = self.get_network(self_type='l', layers = self.layers)        
        

        # 3. Decoder for reconstruct three modalities
        self.decoder_l = nn.Conv1d(self.d_l * 2, self.d_l, kernel_size=1, padding=0, bias=False)     
        self.decoder_v = nn.Conv1d(self.d_v * 2, self.d_v, kernel_size=1, padding=0, bias=False)
        self.decoder_a = nn.Conv1d(self.d_a * 2, self.d_a, kernel_size=1, padding=0, bias=False)

        # for calculate cosine sim between s_x
        self.proj_cosine_l = nn.Linear(combined_dim_low * (self.len_l - args.conv1d_kernel_size_l + 1), combined_dim_low)     
        self.proj_cosine_v = nn.Linear(combined_dim_low * (self.len_v - args.conv1d_kernel_size_v + 1), combined_dim_low)
        self.proj_cosine_a = nn.Linear(combined_dim_low * (self.len_a - args.conv1d_kernel_size_a + 1), combined_dim_low)

        # for align c_l, c_v, c_a
        self.align_c_l = nn.Linear(combined_dim_low * (self.len_l - args.conv1d_kernel_size_l + 1), combined_dim_low)
        self.align_c_v = nn.Linear(combined_dim_low * (self.len_v - args.conv1d_kernel_size_v + 1), combined_dim_low)
        self.align_c_a = nn.Linear(combined_dim_low * (self.len_a - args.conv1d_kernel_size_a + 1), combined_dim_low)

        self.self_attentions_c_l = self.get_network(self_type='l')
        self.self_attentions_c_v = self.get_network(self_type='v')
        self.self_attentions_c_a = self.get_network(self_type='a')

        self.proj1_c = nn.Linear(self.d_l * 3, self.d_l * 3)
        self.proj2_c = nn.Linear(self.d_l * 3, self.d_l * 3)
        self.out_layer_c = nn.Linear(self.d_l * 3, output_dim)


        # 4 Multimodal Crossmodal Attentions
        self.trans_l_with_a = self.get_network(self_type='la', layers = self.layers)  
        self.trans_l_with_v = self.get_network(self_type='lv', layers = self.layers) 
        self.trans_a_with_l = self.get_network(self_type='al')
        self.trans_a_with_v = self.get_network(self_type='av')
        self.trans_v_with_l = self.get_network(self_type='vl')
        self.trans_v_with_a = self.get_network(self_type='va')
        self.trans_l_mem = self.get_network(self_type='l_mem', layers=self.layers)
        self.trans_a_mem = self.get_network(self_type='a_mem', layers=3)
        self.trans_v_mem = self.get_network(self_type='v_mem', layers=3)

        # 初始化外部记忆模块
        self.external_memory_l = ExternalMemory(self.d_l, mem_size=64)
        self.external_memory_a = ExternalMemory(self.d_a, mem_size=64)
        self.external_memory_v = ExternalMemory(self.d_v, mem_size=64)

        # 5. fc layers for shared features
        self.proj1_l_low = nn.Linear(combined_dim_low * (self.len_l - args.conv1d_kernel_size_l + 1), combined_dim_low)
        self.proj2_l_low = nn.Linear(combined_dim_low, combined_dim_low * (self.len_l - args.conv1d_kernel_size_l + 1))
        self.out_layer_l_low = nn.Linear(combined_dim_low * (self.len_l - args.conv1d_kernel_size_l + 1), output_dim)
        self.proj1_v_low = nn.Linear(combined_dim_low * (self.len_v - args.conv1d_kernel_size_v + 1), combined_dim_low)
        self.proj2_v_low = nn.Linear(combined_dim_low, combined_dim_low * (self.len_v - args.conv1d_kernel_size_v + 1))
        self.out_layer_v_low = nn.Linear(combined_dim_low * (self.len_v - args.conv1d_kernel_size_v + 1), output_dim)
        self.proj1_a_low = nn.Linear(combined_dim_low * (self.len_a - args.conv1d_kernel_size_a + 1), combined_dim_low)
        self.proj2_a_low = nn.Linear(combined_dim_low, combined_dim_low * (self.len_a - args.conv1d_kernel_size_a + 1))
        self.out_layer_a_low = nn.Linear(combined_dim_low * (self.len_a - args.conv1d_kernel_size_a + 1), output_dim)

        
        # 6. fc layers for specific features
        self.proj1_l_high = nn.Linear(combined_dim_high, combined_dim_high)
        self.proj2_l_high = nn.Linear(combined_dim_high, combined_dim_high)
        self.out_layer_l_high = nn.Linear(combined_dim_high, output_dim)
        self.proj1_v_high = nn.Linear(combined_dim_high, combined_dim_high)
        self.proj2_v_high = nn.Linear(combined_dim_high, combined_dim_high)
        self.out_layer_v_high = nn.Linear(combined_dim_high, output_dim)
        self.proj1_a_high = nn.Linear(combined_dim_high, combined_dim_high)
        self.proj2_a_high = nn.Linear(combined_dim_high, combined_dim_high)
        self.out_layer_a_high = nn.Linear(combined_dim_high, output_dim)
        
        # 7. project for fusion
        self.projector_l = nn.Linear(self.d_l, self.d_l)
        self.projector_v = nn.Linear(self.d_v, self.d_v)
        self.projector_a = nn.Linear(self.d_a, self.d_a)
        self.projector_c = nn.Linear(3 * self.d_l, 3 * self.d_l)
        
        # 添加记忆融合机制
        self.memory_fusion = nn.Sequential(
            nn.Linear(self.d_l + self.d_a + self.d_v, (self.d_l + self.d_a + self.d_v) // 2),
            nn.ReLU(),
            nn.Linear((self.d_l + self.d_a + self.d_v) // 2, self.d_l)
        )
        
        # 添加跨模态记忆注意力
        self.cross_modal_attn = nn.MultiheadAttention(embed_dim=self.d_l, 
                                                     num_heads=self.num_heads, 
                                                     dropout=self.attn_dropout)
        
        # 8. final project
        self.proj1 = nn.Linear(combined_dim, combined_dim)
        self.proj2 = nn.Linear(combined_dim, combined_dim)
        self.out_layer = nn.Linear(combined_dim, output_dim)
        
    def pretrain_memory(self, text_samples, audio_samples, video_samples, num_epochs=5, batch_size=32, lr=1e-4):
        """
        预训练外部记忆模块，使其能更好地编码和检索模态特征
        
        Args:
            text_samples: 文本样本的特征，形状为 [num_samples, d_l]
            audio_samples: 音频样本的特征，形状为 [num_samples, d_a]
            video_samples: 视频样本的特征，形状为 [num_samples, d_v]
            num_epochs: 预训练的轮数
            batch_size: 批次大小
            lr: 学习率
        """
        # 设置为训练模式
        self.external_memory_l.train()
        self.external_memory_a.train()
        self.external_memory_v.train()
        
        # 创建优化器
        params = list(self.external_memory_l.parameters()) + \
                list(self.external_memory_a.parameters()) + \
                list(self.external_memory_v.parameters())
        optimizer = torch.optim.Adam(params, lr=lr)
        
        # 创建数据加载器
        num_samples = min(len(text_samples), len(audio_samples), len(video_samples))
        indices = torch.randperm(num_samples)
        
        for epoch in range(num_epochs):
            total_loss = 0.0
            num_batches = 0
            
            # 按批次处理数据
            for i in range(0, num_samples, batch_size):
                batch_indices = indices[i:min(i+batch_size, num_samples)]
                
                # 获取批次数据
                text_batch = text_samples[batch_indices].to(next(self.parameters()).device)
                audio_batch = audio_samples[batch_indices].to(next(self.parameters()).device)
                video_batch = video_samples[batch_indices].to(next(self.parameters()).device)
                
                # 前向传播
                em_h_l = self.external_memory_l(text_batch)
                em_h_a = self.external_memory_a(audio_batch)
                em_h_v = self.external_memory_v(video_batch)
                
                # 计算重构损失
                recon_loss_l = F.mse_loss(em_h_l, text_batch)
                recon_loss_a = F.mse_loss(em_h_a, audio_batch)
                recon_loss_v = F.mse_loss(em_h_v, video_batch)
                
                # 计算对比损失（使不同模态的表示更加一致）
                sim_matrix_la = torch.matmul(F.normalize(em_h_l, dim=1), F.normalize(em_h_a, dim=1).transpose(0, 1))
                sim_matrix_lv = torch.matmul(F.normalize(em_h_l, dim=1), F.normalize(em_h_v, dim=1).transpose(0, 1))
                sim_matrix_av = torch.matmul(F.normalize(em_h_a, dim=1), F.normalize(em_h_v, dim=1).transpose(0, 1))
                
                # 对角线上的元素应该接近1（相同样本的不同模态），其他元素应该接近0
                batch_size_actual = len(batch_indices)
                target = torch.eye(batch_size_actual).to(next(self.parameters()).device)
                
                contrast_loss_la = F.mse_loss(sim_matrix_la, target)
                contrast_loss_lv = F.mse_loss(sim_matrix_lv, target)
                contrast_loss_av = F.mse_loss(sim_matrix_av, target)
                
                # 总损失
                loss = (recon_loss_l + recon_loss_a + recon_loss_v) + \
                       0.1 * (contrast_loss_la + contrast_loss_lv + contrast_loss_av)
                
                # 反向传播和优化
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
            
            avg_loss = total_loss / num_batches
            print(f"Memory Pretraining - Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
        
        # 设置回评估模式
        self.external_memory_l.eval()
        self.external_memory_a.eval()
        self.external_memory_v.eval()
        
        print("Memory pretraining completed!")
        
    def initialize_external_memories(self, text_vectors=None, audio_vectors=None, video_vectors=None):
        """
        使用提供的向量初始化外部记忆模块
        
        Args:
            text_vectors: 形状为 [mem_size, d_l] 的张量，用于初始化文本记忆
            audio_vectors: 形状为 [mem_size, d_a] 的张量，用于初始化音频记忆
            video_vectors: 形状为 [mem_size, d_v] 的张量，用于初始化视频记忆
        """
        if text_vectors is not None:
            with torch.no_grad():
                num_vectors = min(text_vectors.size(0), self.external_memory_l.mem_size)
                self.external_memory_l.memory.data[:num_vectors] = text_vectors[:num_vectors]
        if audio_vectors is not None:
            with torch.no_grad():
                num_vectors = min(audio_vectors.size(0), self.external_memory_a.mem_size)
                self.external_memory_a.memory.data[:num_vectors] = audio_vectors[:num_vectors]
        if video_vectors is not None:
            with torch.no_grad():
                num_vectors = min(video_vectors.size(0), self.external_memory_v.mem_size)
                self.external_memory_v.memory.data[:num_vectors] = video_vectors[:num_vectors]
            
    def get_network(self, self_type='l', layers=-1):
        if self_type in ['l', 'al', 'vl']:
            embed_dim, attn_dropout = self.d_l, self.attn_dropout
        elif self_type in ['a', 'la', 'va']:
            embed_dim, attn_dropout = self.d_a, self.attn_dropout_a
        elif self_type in ['v', 'lv', 'av']:
            embed_dim, attn_dropout = self.d_v, self.attn_dropout_v        
        elif self_type == 'l_mem':
            embed_dim, attn_dropout = self.d_l, self.attn_dropout
        elif self_type == 'a_mem':
            embed_dim, attn_dropout = self.d_a, self.attn_dropout
        elif self_type == 'v_mem':
            embed_dim, attn_dropout = self.d_v, self.attn_dropout
        else:
            raise ValueError("Unknown network type")

        return TransformerEncoder(embed_dim=embed_dim,
                                  num_heads=self.num_heads,
                                  layers=max(self.layers, layers),
                                  attn_dropout=attn_dropout,
                                  relu_dropout=self.relu_dropout,
                                  res_dropout=self.res_dropout,
                                  embed_dropout=self.embed_dropout,
                                  attn_mask=self.attn_mask)


    def forward(self, text, audio, video):
        #extraction
        if self.use_bert:
            text = self.text_model(text)
        x_l = F.dropout(text.transpose(1, 2), p=self.text_dropout, training=self.training) 
        x_a = audio.transpose(1, 2)
        x_v = video.transpose(1, 2)
        

        proj_x_l = x_l if self.orig_d_l == self.d_l else self.proj_l(x_l) 
        proj_x_a = x_a if self.orig_d_a == self.d_a else self.proj_a(x_a) 
        proj_x_v = x_v if self.orig_d_v == self.d_v else self.proj_v(x_v)
        
        proj_x_l = proj_x_l.permute(2, 0, 1)   
        proj_x_v = proj_x_v .permute(2, 0, 1)  
        proj_x_a = proj_x_a.permute(2, 0, 1)
        
        # --- 新增：消融实验逻辑 ---
        # 如果设置为‘text_only’，则将视觉和语音特征置为零向量
        if self.ablation_mode == 'text_only':
            # 获取特征向量的维度（假设batch size相同）
            batch_size = proj_x_l.size(0)
            # 创建与visual_feat形状相同的零张量，并放在相同的设备上
            proj_x_v = torch.zeros_like(proj_x_v)
            proj_x_a = torch.zeros_like(proj_x_a)
        elif self.ablation_mode == 'no_text':
            proj_x_l = torch.zeros_like(proj_x_l)
            
        elif self.ablation_mode == 'no_audio':
            proj_x_a = torch.zeros_like(proj_x_a)
            
        elif self.ablation_mode == 'no_visual':
            proj_x_v = torch.zeros_like(proj_x_v)
        
        #disentanglement
        s_l = self.encoder_s_l(proj_x_l)    
        s_v = self.encoder_s_v(proj_x_v)
        s_a = self.encoder_s_a(proj_x_a)

        c_l = self.encoder_c(proj_x_l)
        c_v = self.encoder_c(proj_x_v)
        c_a = self.encoder_c(proj_x_a)


        s_l = s_l.permute(1, 2, 0)   
        s_v = s_v.permute(1, 2, 0)
        s_a = s_a.permute(1, 2, 0)

        c_l = c_l.permute(1, 2, 0)
        c_v = c_v.permute(1, 2, 0)
        c_a = c_a.permute(1, 2, 0)
        c_list = [c_l, c_v, c_a]


        c_l_sim = self.align_c_l(c_l.contiguous().view(x_l.size(0), -1))
        c_v_sim = self.align_c_v(c_v.contiguous().view(x_l.size(0), -1))
        c_a_sim = self.align_c_a(c_a.contiguous().view(x_l.size(0), -1))
        
        recon_l = self.decoder_l(torch.cat([s_l, c_list[0]], dim=1))
        recon_v = self.decoder_v(torch.cat([s_v, c_list[1]], dim=1))
        recon_a = self.decoder_a(torch.cat([s_a, c_list[2]], dim=1))

        recon_l = recon_l.permute(2, 0, 1)  
        recon_v = recon_v.permute(2, 0, 1)   
        recon_a = recon_a.permute(2, 0, 1)

        s_l_r = self.encoder_s_l(recon_l).permute(1, 2, 0)                                                                                             
        s_v_r = self.encoder_s_v(recon_v).permute(1, 2, 0)
        s_a_r = self.encoder_s_a(recon_a).permute(1, 2, 0)
        
        s_l = s_l.permute(2, 0, 1)  
        s_v = s_v.permute(2, 0, 1)   
        s_a = s_a.permute(2, 0, 1)

        c_l = c_l.permute(2, 0, 1)
        c_v = c_v.permute(2, 0, 1)
        c_a = c_a.permute(2, 0, 1)
       
       #enhancement
        hs_l_low = c_l.transpose(0, 1).contiguous().view(x_l.size(0), -1)  
        #hs_l_low = proj_x_l.transpose(0, 1).contiguous().view(x_l.size(0), -1)
        repr_l_low = self.proj1_l_low(hs_l_low)                            
        hs_proj_l_low = self.proj2_l_low(
            F.dropout(F.relu(repr_l_low, inplace=True), p=self.output_dropout, training=self.training))
        hs_proj_l_low += hs_l_low         
        logits_l_low = self.out_layer_l_low(hs_proj_l_low)

        hs_v_low = c_v.transpose(0, 1).contiguous().view(x_v.size(0), -1)
        #hs_v_low = proj_x_v.transpose(0, 1).contiguous().view(x_v.size(0), -1)
        repr_v_low = self.proj1_v_low(hs_v_low)
        hs_proj_v_low = self.proj2_v_low(
            F.dropout(F.relu(repr_v_low, inplace=True), p=self.output_dropout, training=self.training))
        hs_proj_v_low += hs_v_low
        logits_v_low = self.out_layer_v_low(hs_proj_v_low)

        hs_a_low = c_a.transpose(0, 1).contiguous().view(x_a.size(0), -1)
        #hs_a_low = proj_x_a.transpose(0, 1).contiguous().view(x_a.size(0), -1)
        repr_a_low = self.proj1_a_low(hs_a_low)
        hs_proj_a_low = self.proj2_a_low(
            F.dropout(F.relu(repr_a_low, inplace=True), p=self.output_dropout, training=self.training))
        hs_proj_a_low += hs_a_low
        logits_a_low = self.out_layer_a_low(hs_proj_a_low)

        c_l_att = self.self_attentions_c_l(c_l)
        #c_l_att = self.self_attentions_c_l(proj_x_l)  
        if type(c_l_att) == tuple:
            c_l_att = c_l_att[0]
        c_l_att = c_l_att[-1]

        c_v_att = self.self_attentions_c_v(c_v)
        #c_v_att = self.self_attentions_c_v(proj_x_v)
        if type(c_v_att) == tuple:
            c_v_att = c_v_att[0]
        c_v_att = c_v_att[-1]

        c_a_att = self.self_attentions_c_a(c_a)
        #c_a_att = self.self_attentions_c_a(proj_x_a)
        if type(c_a_att) == tuple:
            c_a_att = c_a_att[0]
        c_a_att = c_a_att[-1]

        c_fusion = torch.cat([c_l_att, c_v_att, c_a_att], dim=1)   

        c_proj = self.proj2_c(
            F.dropout(F.relu(self.proj1_c(c_fusion), inplace=True), p=self.output_dropout,
                      training=self.training))
        c_proj += c_fusion                       
        logits_c = self.out_layer_c(c_proj)     
        
        # LFA
        # L --> L                
        #h_ls = s_l
        h_ls = c_l
        h_ls = self.trans_l_mem(h_ls)
        if type(h_ls) == tuple:
            h_ls = h_ls[0]
        last_h_l = last_hs = h_ls[-1]

        # A --> L
        #h_l_with_as = self.trans_l_with_a(s_l, s_a, s_a)
        h_l_with_as = self.trans_l_with_a(c_l, c_a, c_a)
        h_as = h_l_with_as
        h_as = self.trans_a_mem(h_as)
        if type(h_as) == tuple:
            h_as = h_as[0]
        last_h_a = last_hs = h_as[-1]

        # V --> L
        #h_l_with_vs = self.trans_l_with_v(s_l, s_v, s_v)
        h_l_with_vs = self.trans_l_with_v(c_l, c_v, c_v)
        h_vs = h_l_with_vs
        h_vs = self.trans_v_mem(h_vs)
        if type(h_vs) == tuple:
            h_vs = h_vs[0]
        last_h_v = last_hs = h_vs[-1]

        # 对各模态应用外部记忆模块
        em_h_l = self.external_memory_l(last_h_l)
        em_h_a = self.external_memory_a(last_h_a)
        em_h_v = self.external_memory_v(last_h_v)
        
        # 融合不同模态的记忆信息
        memory_concat = torch.cat([em_h_l, em_h_a, em_h_v], dim=-1)
        memory_fused = self.memory_fusion(memory_concat)
        
        # 应用跨模态注意力机制
        # 将 em_h_l 作为查询，[em_h_l, em_h_a, em_h_v] 作为键值对
        memory_stacked = torch.stack([em_h_l, em_h_a, em_h_v], dim=0)  # [3, batch_size, dim]
        cross_attn_out, _ = self.cross_modal_attn(
            em_h_l.unsqueeze(0),  # 查询: [1, batch_size, dim]
            memory_stacked,       # 键: [3, batch_size, dim]
            memory_stacked        # 值: [3, batch_size, dim]
        )
        cross_attn_out = cross_attn_out.squeeze(0)  # [batch_size, dim]
        
        # 结合原始记忆和跨模态增强的记忆
        last_h_l = em_h_l + 0.5 * memory_fused + 0.5 * cross_attn_out
        last_h_a = em_h_a
        last_h_v = em_h_v

        hs_proj_l_high = self.proj2_l_high(
            F.dropout(F.relu(self.proj1_l_high(last_h_l), inplace=True), p=self.output_dropout, training=self.training))
        hs_proj_l_high += last_h_l
        logits_l_high = self.out_layer_l_high(hs_proj_l_high)

        hs_proj_v_high = self.proj2_v_high(
            F.dropout(F.relu(self.proj1_v_high(last_h_v), inplace=True), p=self.output_dropout, training=self.training))
        hs_proj_v_high += last_h_v
        logits_v_high = self.out_layer_v_high(hs_proj_v_high)

        hs_proj_a_high = self.proj2_a_high(
            F.dropout(F.relu(self.proj1_a_high(last_h_a), inplace=True), p=self.output_dropout,
                      training=self.training))
        hs_proj_a_high += last_h_a
        logits_a_high = self.out_layer_a_high(hs_proj_a_high)
        
        #fusion
        last_h_l = torch.sigmoid(self.projector_l(hs_proj_l_high))   
        last_h_v = torch.sigmoid(self.projector_v(hs_proj_v_high))
        last_h_a = torch.sigmoid(self.projector_a(hs_proj_a_high))
        c_fusion = torch.sigmoid(self.projector_c(c_fusion))
        
        last_hs = torch.cat([last_h_l, last_h_v, last_h_a, c_fusion], dim=1)   

        #prediction
        last_hs_proj = self.proj2(
            F.dropout(F.relu(self.proj1(last_hs), inplace=True), p=self.output_dropout, training=self.training))
        last_hs_proj += last_hs                          

        output = self.out_layer(last_hs_proj)

        res = {
            'origin_l': proj_x_l,
            'origin_v': proj_x_v,
            'origin_a': proj_x_a,
            's_l': s_l,        
            's_v': s_v,
            's_a': s_a,
            'c_l': c_l,
            'c_v': c_v,
            'c_a': c_a,
            's_l_r': s_l_r,
            's_v_r': s_v_r,
            's_a_r': s_a_r,
            'recon_l': recon_l,
            'recon_v': recon_v,
            'recon_a': recon_a,
            'c_l_sim': c_l_sim,
            'c_v_sim': c_v_sim,
            'c_a_sim': c_a_sim,
            'logits_l_hetero': logits_l_high,           
            'logits_v_hetero': logits_v_high, 
            'logits_a_hetero': logits_a_high,
            'logits_c': logits_c,
            'output_logit': output
        }
        return res
