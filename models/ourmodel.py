import torch
import torch.nn as nn
#test
from PIL import Image
import torch.nn.functional as F
from torch_scatter import scatter_add
from torch_geometric.utils import softmax
import dgl
from transformers import AutoTokenizer, AutoModel
import ipdb
import numpy as np
import torch
import torch.nn as nn
from torch import nn
import pandas as pd
import os
from models.llava.model.builder import load_pretrained_model
from models.llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
from PIL import Image
from models.llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_STRAT_PROMPTS
from models.layers.cross_attention import FeedForward, MMAttentionLayer

# class fusion_fc(torch.nn.Module):
#     def __init__(self):
#         super(fusion_fc, self).__init__()
#         self.gate_nn = gate_nn
#         self.nn = nn



#     def forward(self, x, batch, size=None):
#         """"""
#         x = x.unsqueeze(-1) if x.dim() == 1 else x
#         #size是说明有几类节点
#         #此处是就一种，所以是一个全零的张量，代表了每个节点的类型，或是说明每个节点都属于同一张图
#         #batch = torch.zeros(len(x_img),dtype=torch.long)
#         size = batch[-1].item() + 1 if size is None else size
#         assert gate.dim() == x.dim() and gate.size(0) == x.size(0)
#         #[1,2,1]做这个图专门的softmax，如果第二个参数是[0,1,0]，就会变成[0.5,2,0.5]
#         gate = softmax(x, batch, num_nodes=size)
#         #将每个target node的与其邻接节点的边的权重之求和，最终得到的输出维度是节点数目
#         #scatter_add([1,1,2],[2,0,0])=[3,0,1] --- 第二个向量代表第一个向量里面每个数字需要加到哪个位置上面
#         #out是所有节点的总特征
#         out = scatter_add(gate, batch, dim=0, dim_size=size)
#         '''
#         print((gate).shape)
#         print((x).shape)
#         print((gate * x).shape)
#         print(out.shape)
#         torch.Size([237, 1])
#         torch.Size([237, 512])
#         torch.Size([237, 512])
#         torch.Size([1, 512])
#         237的部分是看他是不是图片,图片大小才回浮动,其他模态都是5或6
#         '''
#         return out,gate


# def GNN_relu_Block(dim2, dropout=0.3):
#     r"""
#     Multilayer Reception Block w/ Self-Normalization (Linear + ELU + Alpha Dropout)
#     args:
#         dim1 (int): Dimension of input features
#         dim2 (int): Dimension of output features
#         dropout (float): Dropout rate
#     """
#     return nn.Sequential(
# #             GATConv(in_channels=dim1,out_channels=dim2),
#             nn.ReLU(),
#             LayerNorm(dim2),
#             nn.Dropout(p=dropout))

class CrossAttention(nn.Module):
    def __init__(self, in_dim1, in_dim2, k_dim, v_dim, num_heads):
        super(CrossAttention, self).__init__()
        self.num_heads = num_heads
        self.k_dim = k_dim
        self.v_dim = v_dim
        
        self.proj_q1 = nn.Linear(in_dim1, k_dim * num_heads, bias=False)
        self.proj_k2 = nn.Linear(in_dim2, k_dim * num_heads, bias=False)
        self.proj_v2 = nn.Linear(in_dim2, v_dim * num_heads, bias=False)
        self.proj_o = nn.Linear(v_dim * num_heads, in_dim1)
        
    def forward(self, x1, x2, mask=None):
        batch_size, seq_len1, in_dim1 = x1.size()
        seq_len2 = x2.size(1)
        
        q1 = self.proj_q1(x1).view(batch_size, seq_len1, self.num_heads, self.k_dim).permute(0, 2, 1, 3)
        k2 = self.proj_k2(x2).view(batch_size, seq_len2, self.num_heads, self.k_dim).permute(0, 2, 3, 1)
        v2 = self.proj_v2(x2).view(batch_size, seq_len2, self.num_heads, self.v_dim).permute(0, 2, 1, 3)
        
        attn = torch.matmul(q1, k2) / self.k_dim**0.5
        
        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)
        
        attn = F.softmax(attn, dim=-1)
        output = torch.matmul(attn, v2).permute(0, 2, 1, 3).contiguous().view(batch_size, seq_len1, -1)
        output = self.proj_o(output)
        
        return output


# 定义GAT神经层
class GATLayer(nn.Module):
    def __init__(self, g, in_dim, out_dim):
        super(GATLayer, self).__init__()
        # 数据
        self.g = g
        # 对应公式中1的 W，用于特征的线性变换
        self.fc = nn.Linear(in_dim, out_dim, bias=False)
        # 对应公式2中的 a, 输入拼接的zi和zj（2 * out_dim），输出eij（一个数值）
        self.attn_fc = nn.Linear(2 * out_dim, 1, bias=False)

        # 初始化参数
        self.reset_parameters()

    def reset_parameters(self):
        # 随机初始化需要学习的参数
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.fc.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_fc.weight, gain=gain)

    def edge_attention(self, edges):
        # 对应公式2中的拼接操作，即zi || zj
        z2 = torch.cat([edges.src['z'], edges.dst['z']], dim=1)
        # 拼接之后对应公式2中激活函数里的计算操作，即a(zi || zj)
        a = self.attn_fc(z2)
        # 算出来的值经过leakyReLU激活得到eij,保存在e变量中
        return {'e': F.leaky_relu(a)}

    def message_func(self, edges):
        # 汇聚信息，传递之前计算好的z（对应节点的特征） 和 e
        return {'z': edges.src['z'], 'e': edges.data['e']}

    def reduce_func(self, nodes):
        # 对应公式3，eij们经过softmax即可得到特征的权重αij
        alpha = F.softmax(nodes.mailbox['e'], dim=1)
        # 计算出权重之后即可通过 权重αij * 变换后的特征zj 求和计算出节点更新后的特征
        # 不过激活函数并不在这里，代码后面有用到ELU激活函数
        h = torch.sum(alpha * nodes.mailbox['z'], dim=1)
        return {'h': h}

    # 正向传播方式
    def forward(self, h):   
        # 对应公式1，先转换特征
        z = self.fc(h)
        # 将转换好的特征保存在z
        self.g.ndata['z'] = z
        # 对应公式2，得出e
        self.g.apply_edges(self.edge_attention)
        # 对应公式3、4计算出注意力权重α并且得出最后的hi
        self.g.update_all(self.message_func, self.reduce_func)
        # 返回并清除hi
        return self.g.ndata.pop('h')

# 定义多头注意力机制的GAT层
class MultiHeadGATLayer(nn.Module):
    def __init__(self, g, in_dim, out_dim, num_heads, merge='cat'):
        super(MultiHeadGATLayer, self).__init__()
        # 多头注意力机制的头数（注意力机制的数量）
        self.heads = nn.ModuleList()
        # 添加对应的注意力机制层，即GAT神经层
        for i in range(num_heads):
            self.heads.append(GATLayer(g, in_dim, out_dim))
        self.merge = merge  # 使用拼接的方法，否则取平均

    def forward(self, h):
        # 获取每套注意力机制得到的hi
        head_outs = [attn_head(h) for attn_head in self.heads]
        if self.merge == 'cat':
            # 每套的hi拼接
            return torch.cat(head_outs, dim=1)
        else:
            # 所有的hi对应元素求平均
            return torch.mean(torch.stack(head_outs))
        



# 定义GAT模型
class GATA(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim, patch_count,global_act,num_heads=1,):
        super(GATA, self).__init__()
        self.global_act = global_act
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        if global_act is True:

            self.cross_attender = CrossAttention(in_dim1=in_dim, in_dim2=in_dim, k_dim=64, v_dim=64, num_heads=8)
            self.feed_forward = FeedForward(in_dim, dropout=0.1)
            self.layer_norm = nn.LayerNorm(in_dim)
            self.fc = nn.Sequential(
                nn.Linear(in_dim, int(in_dim/4)),
                nn.ReLU(),
                nn.Linear(int(in_dim/4), out_dim)
            )
            # self.fc = nn.Sequential(
            #     nn.Linear(in_dim*(patch_count+1), int(in_dim*(patch_count+1)/4)),
            #     nn.Linear(int(in_dim*(patch_count+1)/4), int(in_dim*(patch_count+1)/4)),
            #     nn.ReLU(),
            #     nn.Linear(int(in_dim*(patch_count+1)/4), out_dim),
            # )
        else:
            edges = [(i, j) for i in range(patch_count+1) for j in range(patch_count+1) if i != j]
            g = dgl.graph(edges, num_nodes=patch_count+1)
            g = g.to(self.device)
            self.layer1 = MultiHeadGATLayer(g, self.in_dim, self.hidden_dim, self.num_heads).to(self.device)
        # 这里需要注意的是，因为第一层多头注意力机制层layer1选择的是拼接
        # 那么传入第二层的参数应该是第一层的 输出维度 * 头数
            self.layer2 = MultiHeadGATLayer(g, self.hidden_dim * self.num_heads, self.in_dim, 1).to(self.device)
            self.fc = nn.Sequential(
                nn.Linear(in_dim, int(in_dim/4)),
                nn.ReLU(),
                nn.Linear(int(in_dim/4), out_dim)
            ).to(self.device)
        

        

    def forward(self,pi_total_vector):
 

        if self.global_act is not True:
            batch = torch.zeros(len(pi_total_vector),dtype=torch.long).to(self.device)
            pi_total_vector = self.layer1(pi_total_vector)
            # ELU激活函数
            pi_total_vector = F.elu(pi_total_vector)
            pi_total_vector = self.layer2(pi_total_vector)
            pi_total_vector = pi_total_vector.unsqueeze(-1) if pi_total_vector.dim() == 1 else pi_total_vector
            #size是说明有几类节点
            #此处是就一种，所以是一个全零的张量，代表了每个节点的类型，或是说明每个节点都属于同一张图
            #batch = torch.zeros(len(x_img),dtype=torch.long)

            size = batch[-1].item() + 1
            
            # assert gate.dim() == x.dim() and gate.size(0) == x.size(0)
            #[1,2,1]做这个图专门的softmax，如果第二个参数是[0,1,0]，就会变成[0.5,2,0.5]
            pi_total_vector = softmax(pi_total_vector, batch, num_nodes=size)
            #将每个target node的与其邻接节点的边的权重之求和，最终得到的输出维度是节点数目
            #scatter_add([1,1,2],[2,0,0])=[3,0,1] --- 第二个向量代表第一个向量里面每个数字需要加到哪个位置上面
            #out是所有节点的总特征
            #聚合版本(输出(1,out_dim))

            pi_total_vector = self.fc(pi_total_vector)
            out = scatter_add(pi_total_vector, batch, dim=0, dim_size=size)
            
        
        else:
            pi_global_vector = pi_total_vector[:, -1:, :]  # 选择最后一个时间步的向量
            # 获取剩下的向量，形状变为 (bs, sq_len-1, emb_dim)
            pi_remaining_vectors = pi_total_vector[:, :-1, :]  # 获取除了最后一个的所有向量
            pi_total_vector = self.cross_attender(pi_global_vector ,pi_remaining_vectors).permute(1, 0, 2) 
            pi_total_vector = self.feed_forward(pi_total_vector)
            pi_total_vector = self.layer_norm(pi_total_vector)
            out = self.fc(pi_total_vector)
            #---> aggregate 
        return out
        
        # else:
        #     pi_global_vector = pi_total_vector[-1].unsqueeze(0)  # 取出最后一个 (1, 768) 向量
        #     pi_remaining_vectors = pi_total_vector[:-1]  
        #     batch = torch.zeros(len(pi_remaining_vectors),dtype=torch.long).to(self.device)
        #     pi_total_vector = self.layer1(pi_remaining_vectors)
        #     # ELU激活函数
        #     pi_total_vector = F.elu(pi_total_vector)
        #     pi_total_vector = self.layer2(pi_total_vector)
        #     pi_total_vector = pi_total_vector.unsqueeze(-1) if pi_total_vector.dim() == 1 else pi_total_vector
        #     #size是说明有几类节点
        #     #此处是就一种，所以是一个全零的张量，代表了每个节点的类型，或是说明每个节点都属于同一张图
        #     #batch = torch.zeros(len(x_img),dtype=torch.long)

        #     size = batch[-1].item() + 1
            
        #     # assert gate.dim() == x.dim() and gate.size(0) == x.size(0)
        #     #[1,2,1]做这个图专门的softmax，如果第二个参数是[0,1,0]，就会变成[0.5,2,0.5]
        #     pi_total_vector = softmax(pi_total_vector, batch, num_nodes=size)
        #     #将每个target node的与其邻接节点的边的权重之求和，最终得到的输出维度是节点数目
        #     #scatter_add([1,1,2],[2,0,0])=[3,0,1] --- 第二个向量代表第一个向量里面每个数字需要加到哪个位置上面
        #     #out是所有节点的总特征
        #     #聚合版本(输出(1,out_dim))

        #     pi_total_vector = self.fc(pi_total_vector+pi_global_vector)
        #     out = scatter_add(pi_total_vector, batch, dim=0, dim_size=size)

        #     return out

    
    #聚合版本
    # def forward(self,pi_total_vector,batch):
    #     # 提取第 5 个元素 (索引为 4)
    #     pi_global_vector = pi_total_vector[4]  # (768,)

    #     # 从原始张量中移除第 5 个元素
    #     pi_total_vector = torch.cat((pi_total_vector[:4], pi_total_vector[5:]), dim=0)  # 现在是 (4, 768)
    #     edges = [(i, j) for i in range(4) for j in range(4) if i != j]
    #     g = dgl.graph(edges, num_nodes=4)
    #     g = g.to(self.device)
    #     layer1 = MultiHeadGATLayer(g, self.in_dim, self.hidden_dim, self.num_heads).to(self.device)
    #     # 这里需要注意的是，因为第一层多头注意力机制层layer1选择的是拼接
    #     # 那么传入第二层的参数应该是第一层的 输出维度 * 头数
    #     layer2 = MultiHeadGATLayer(g, self.hidden_dim * self.num_heads, self.in_dim, 1).to(self.device)
    #     pi_total_vector = layer1(pi_total_vector)
    #     # ELU激活函数
    #     pi_total_vector = F.elu(pi_total_vector)
    #     pi_total_vector = layer2(pi_total_vector)
    #     pi_total_vector = pi_total_vector.unsqueeze(-1) if pi_total_vector.dim() == 1 else pi_total_vector
    #     #size是说明有几类节点
    #     #此处是就一种，所以是一个全零的张量，代表了每个节点的类型，或是说明每个节点都属于同一张图
    #     #batch = torch.zeros(len(x_img),dtype=torch.long)

    #     size = batch[-1].item() + 1
        
    #     # assert gate.dim() == x.dim() and gate.size(0) == x.size(0)
    #     #[1,2,1]做这个图专门的softmax，如果第二个参数是[0,1,0]，就会变成[0.5,2,0.5]
    #     gate = softmax(pi_total_vector, batch, num_nodes=size)
    #     #将每个target node的与其邻接节点的边的权重之求和，最终得到的输出维度是节点数目
    #     #scatter_add([1,1,2],[2,0,0])=[3,0,1] --- 第二个向量代表第一个向量里面每个数字需要加到哪个位置上面
    #     #out是所有节点的总特征
    #     #聚合版本(输出(1,out_dim))
    #     out = scatter_add(gate, batch, dim=0, dim_size=size).unsqueeze(0)
    #     pi_total_vector = torch.cat((out, pi_global_vector.unsqueeze(0).unsqueeze(0)), dim=1)
    #     pi_total_vector = self.fc(pi_total_vector)
    #     pi_total_vector = torch.mean(pi_total_vector, dim=1)
        
    #     return pi_total_vector


# class PromptEncodeFactory(nn.Module):
#   def __init__(self,in_dim,encoder_model="dmis-lab/biobert-base-cased-v1.2",learnable_n=40,patch_count = 4,):
#     super().__init__()
    
#     self.model = AutoModel.from_pretrained(encoder_model)
#     for param in self.model.encoder.parameters():
#         param.requires_grad = False  # 冻结 encoder 的参数
#     self.tokenizer = AutoTokenizer.from_pretrained(encoder_model)
#     self.embeddings = self.model.embeddings
#     for param in self.model.parameters():
#         param.requires_grad = False
#     for param in self.embeddings.parameters():
#         param.requires_grad = False
#     self.learnable_n = learnable_n
#     self.learnable_prompts = nn.ParameterList(
#             [nn.Parameter(torch.empty((1, self.learnable_n, in_dim), dtype=torch.float32)) for _ in range(patch_count+1)]
#         )
        
#         # 初始化每个可学习的 prompt 向量
#     for prompt in self.learnable_prompts:
#             nn.init.normal_(prompt)


#   def forward(self,texts=[]):
#     prompt_t = self.tokenizer(texts, truncation=True, padding=True,return_tensors="pt")  # 将文本转为张量
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     token_type_ids = prompt_t.get('token_type_ids').to(device)  # 可选  
    
#     input_ids = prompt_t['input_ids'].to(device)


#     embeds = self.embeddings(input_ids=input_ids, token_type_ids=token_type_ids)



#     outputs = []  # 初始化一个空列表以存储所有的 output

#     for index, text in enumerate(texts):
#         learnable_prompt_count = min((embeds[index].shape[0] // 10), self.learnable_n)  # 计算要使用的 learnable prompts 数量
#         learnable_prompt = self.learnable_prompts[index].to(device)
#         text_with_learnable_prompt_vector = torch.cat([learnable_prompt, embeds[index].unsqueeze(0)], dim=1)
#         # text_with_learnable_prompt_vector = torch.cat([embeds[index].unsqueeze(0)], dim=1)
#         # 构造 attention_mask
#         attention_mask = torch.ones(text_with_learnable_prompt_vector.size()[:-1]).to(device)
#         attention_mask[learnable_prompt_count+1:] = 0  # 将未使用的 learnable prompts 的部分设为 0
        
#         output = self.model.encoder(text_with_learnable_prompt_vector, attention_mask=attention_mask)
#         output = output.last_hidden_state
#         cls_embedding = output[:, 0, :]
        
#         outputs.append(cls_embedding)  # 将每个 output 的第一个维度添加到 outputs 列表中

#     outputs = torch.cat(outputs, dim=0)  # 在新的维度上汇总
#     return outputs
  

class PromptEncodeFactory(nn.Module):
  def __init__(self, in_dim, encoder_model="dmis-lab/biobert-base-cased-v1.2", learnable_n=35, patch_count=4,batch_size=1,max_token_len = 411):
    super().__init__()
    self.patch_count = patch_count
    self.max_token_len = max_token_len
    # 加载预训练的模型
    self.model = AutoModel.from_pretrained(encoder_model)
    for param in self.model.encoder.parameters():
        param.requires_grad = False  # 冻结 encoder 的参数
    self.tokenizer = AutoTokenizer.from_pretrained(encoder_model)
    self.embeddings = self.model.embeddings
    for param in self.model.parameters():
        param.requires_grad = False
    for param in self.embeddings.parameters():
        param.requires_grad = False
    self.batch_size = batch_size
    self.learnable_n = learnable_n
    # 初始化一个矩阵来存储所有 learnable prompts
    self.learnable_prompts = nn.Parameter(torch.empty(((patch_count + 1)*batch_size,learnable_n, in_dim), dtype=torch.float32))
    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 初始化 learnable prompts
    nn.init.normal_(self.learnable_prompts)

  def pro(self,attn_mask):
      
    # 统计每个行向量中1的个数
    counts = attn_mask.sum(dim=1)  # 形状 (25,)


    # 计算每个行要设置为1的数量
    num_ones = torch.clamp(counts // 15, max=self.learnable_n) # 形状 (25,)

    # 创建目标张量 (25, 40)，全零初始化
    lp_mask = torch.zeros(self.batch_size*(self.patch_count+1), self.learnable_n, dtype=torch.int).to(self.device) 

    # 创建一个范围向量，用于比较每行的num_ones数量
    range_vector = torch.arange(self.learnable_n).unsqueeze(0).to(self.device)   # 形状 (1, 40)

    # 生成填充结果：每行中前 num_ones[i] 个位置为1
    lp_mask = (range_vector < num_ones.unsqueeze(1)).int()
    return lp_mask

  def forward(self, texts=[]):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    batch_size, num_sentences = len(texts),len(texts[0])
    flat_texts = texts.flatten().tolist()
    prompt_t = self.tokenizer(flat_texts, truncation=True, padding=True, return_tensors="pt")  # 将文本转为张量
    attn_emb_mask = prompt_t.get('attention_mask').to(device)  # 可选
    input_ids = prompt_t['input_ids'].to(device)

    # 获取文本嵌入
    embeds = self.embeddings(input_ids=input_ids)
    embeds = embeds[:, :self.max_token_len, :]
    #(batch_size*num_sen,token_len,emb_dim)
    # 计算 learnable prompt 数量，确保不超过 max 数量
    lp_mask = self.pro(attn_emb_mask)

    text_with_prompts = torch.cat([embeds,self.learnable_prompts ], dim=1)  # 拼接 prompts 和文本嵌入

    attn_mask = torch.cat([attn_emb_mask,lp_mask],dim=1)
    # 将整个 batch 一起输入到 encoder 中
    output = self.model(inputs_embeds=text_with_prompts,
                                attention_mask=attn_mask)

    output = output.last_hidden_state
    cls_embedding = output[:, 0, :]  # 取 [CLS] token 对应的嵌入
    cls_embedding = cls_embedding.view(batch_size, self.patch_count + 1, cls_embedding.shape[-1])
    #(batch_size*patch_count+1,768)
    return cls_embedding



class QLlava(nn.Module):
    def __init__(self,
                 model_path,
                 prompt_base="Considering the clinical information provided, could you give a concise description of the histopathology image shown?",
                 model_base=None,
                 load_8bit = False,
                 load_4bit=True):
        super().__init__()
        model_name = get_model_name_from_path(model_path)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer, self.model, self.image_processor, _ = load_pretrained_model(model_path, model_base, model_name, load_8bit,load_4bit,device=self.device)
        self.prompt_base = prompt_base


    def img2tensor_prompt(self,img,cli=""):
                image_tensors = process_images(img, self.image_processor)

                if type(image_tensors) is list:
                    image_tensor = [image.to(self.model.device, dtype=torch.float16) for image in image_tensors]
                    print("多图并行功能尚未完成")
                    print(sb)
                else:
                    image_tensor = image_tensors.to(self.model.device, dtype=torch.float16)

                prompt = DEFAULT_STRAT_PROMPTS + DEFAULT_IMAGE_TOKEN + '\n' + cli + self.prompt_base
                input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
                return image_tensor,input_ids

        #待实现
    def forward(self,img_total,cli=None):

            image_tensor,input_ids = self.img2tensor_prompt(img_total,cli)
            with torch.inference_mode():
                output_ids = self.model.generate(
                    input_ids,
                    images=image_tensor,
                    do_sample=True,
                    temperature=0.2,
                    max_new_tokens=512,
                    streamer=None,
                    use_cache=True,
                    stopping_criteria=None
                    )
            outputs = self.tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
            return outputs


        #把img切割成patch_count个patch
        #实现patch_count个patch分别加上一个固定的询问prompt进到llava得到每个patch对应一个answar
        #img_g加上结合clinical的prompt进到llava得到answar_g


# class Align(nn.Module):
#     def __init__(self,dim1,dim2):
#         super().__init__()
    
#     def forward(self,t_emb):
#         return emb_a
    
class AlignBlock(nn.Module):
    def __init__(self, embed_dim=768, hidden_dim=128):
        super(AlignBlock, self).__init__()
        self.layernorm = torch.nn.LayerNorm(768) 
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=4,
                dim_feedforward=1024,  # 前馈神经网络中的隐藏层维度
                dropout=0.1,
                activation='relu',
                batch_first=True
            )
        ,num_layers=3
        ,norm=self.layernorm
        )
        self.fc = nn.Linear(embed_dim, hidden_dim)
        self.activation = nn.ReLU()

    def forward(self, x):
        # x: shape (5, 768)
        x_all = x[-1:].unsqueeze(0)  # 提取最后一个向量，形状为 (1, 1, 768)
        x = x[:-1].unsqueeze(0)  # 提取剩余的 (4, 768)，并调整为 (1, 4, 768)

        x = self.transformer_encoder(x) #(5,768)
        x = torch.mean(x, dim=1) # (1, 768)，取平均
        combined = torch.cat((x, x_all.squeeze(0)), dim=0)
        x = self.fc(combined)  # (1, hidden_dim)(1,256)
        x = torch.mean(x, dim=1)
        x = self.activation(x)  # 激活函数
        return x

#完整模型
class CPKSModel(nn.Module):
    def __init__(self,
                 model_path='wisdomik/Quilt-Llava-v1.5-7b',
                 encoder_model_path="dmis-lab/biobert-v1.1",
                 patch_count=4,
                 learnable_n=15,
                 in_dim = 768,
                 out_dim = 128,
                 hidden_dim = 1024,
                 global_act = True,
                 batch_size = 1
                 ):
        super().__init__()
        #待实现
        self.num_patches = patch_count

        self.qllava = QLlava(model_path)
        for param in self.qllava.parameters():
            param.requires_grad = False
        self.promptLearner = PromptEncodeFactory(encoder_model=encoder_model_path,learnable_n=learnable_n,in_dim = in_dim,patch_count=patch_count,batch_size=batch_size)
        # self.align = AlignBlock()
        self.align = GATA(in_dim,out_dim,hidden_dim, patch_count, global_act,num_heads=1,)
    def crop_image_patches(self,folder_name,slide_id, num_patches):
        # 检查 num_patches 是否为正偶数
        if not isinstance(num_patches, int) or num_patches <= 0 or num_patches % 2 != 0:
            raise ValueError("num_patches must be a positive even integer.")
        image_path = os.path.join(folder_name, slide_id)
        # 加载图片
        image = Image.open(image_path)
        width, height = image.size
        # 计算每个 patch 的宽度和高度
        patch_width = width // (num_patches/2)
        patch_height = height // (num_patches/2)

        patches = []

        for i in range(int(num_patches/2)):
            for j in range(int(num_patches/2)):
                # 计算裁剪区域
                left = j * patch_width
                upper = i * patch_height
                right = left + patch_width
                lower = upper + patch_height

                # 裁剪图片

                patch = image.crop((left, upper, right, lower))
                patches.append(patch)
        patches.append(image)
        return patches
    

    def forward(self,data):
        if isinstance(data, tuple):
            slide_id_jpg,clinical_info = data
            ans_total = []  # 初始化空数组
            img_patchs = self.crop_image_patches('/home/ubuntu/disk1/wys/SurvPath/imgs/hnsc_jpgs', slide_id_jpg,self.num_patches)
            for index, img in enumerate(img_patchs):

                ans = self.qllava(img, clinical_info)

                
                ans_total.append(ans)  # 将每个ans添加到ans_total
            ans_total = [s.replace('</s>', '').replace('\n', '') for s in ans_total]
        
        def clean_string(s):
            return s.replace("'", "").replace("[", "").replace("]", "").replace('"', "")

        ans_total = np.vectorize(clean_string)(data)


        pi_total_vector = self.promptLearner(ans_total)

        
        pi_total_vector = self.align(pi_total_vector)
        return pi_total_vector
    

#完整模型
class CPKSModel1(nn.Module):
    def __init__(self,
                 model_path='wisdomik/Quilt-Llava-v1.5-7b',
                 encoder_model_path="dmis-lab/biobert-v1.1",
                 patch_count=4,
                 learnable_n=15,
                 in_dim = 768,
                 out_dim = 128,
                 hidden_dim = 1024,
                 ):
        super().__init__()
        #待实现
        self.num_patches = patch_count

        self.qllava = QLlava(model_path)
        for param in self.qllava.parameters():
            param.requires_grad = False
        self.promptLearner = PromptEncodeFactory(encoder_model=encoder_model_path,learnable_n=learnable_n)
        # self.align = AlignBlock()
        self.align = GATA(in_dim,out_dim,hidden_dim,  num_heads=1)
    def crop_image_patches(self,folder_name,slide_id, num_patches):
        # 检查 num_patches 是否为正偶数
        if not isinstance(num_patches, int) or num_patches <= 0 or num_patches % 2 != 0:
            raise ValueError("num_patches must be a positive even integer.")
        image_path = os.path.join(folder_name, slide_id)
        # 加载图片
        image = Image.open(image_path)
        width, height = image.size
        # 计算每个 patch 的宽度和高度
        patch_width = width // (num_patches/2)
        patch_height = height // (num_patches/2)

        patches = []

        for i in range(int(num_patches/2)):
            for j in range(int(num_patches/2)):
                # 计算裁剪区域
                left = j * patch_width
                upper = i * patch_height
                right = left + patch_width
                lower = upper + patch_height

                # 裁剪图片

                patch = image.crop((left, upper, right, lower))
                patches.append(patch)
        patches.append(image)
        return patches
    

    def forward(self,data):
        if isinstance(data, tuple):
            slide_id_jpg,clinical_info = data
            ans_total = []  # 初始化空数组
            img_patchs = self.crop_image_patches('/home/ubuntu/disk1/wys/SurvPath/imgs/hnsc_jpgs', slide_id_jpg,self.num_patches)
            for index, img in enumerate(img_patchs):

                ans = self.qllava(img, clinical_info)

                
                ans_total.append(ans)  # 将每个ans添加到ans_total
            ans_total = [s.replace('</s>', '').replace('\n', '') for s in ans_total]
        
        ans_total = [s.replace("'", "").replace("[", "").replace("]","").replace('"',"") for s in data]
        pi_total_vector = self.promptLearner(ans_total)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        batch = torch.zeros(len(pi_total_vector),dtype=torch.long).to(device)
        pi_total_vector = self.align(pi_total_vector,batch)
        return pi_total_vector


    
#处理第一阶段--根据提示输出描述
class Qllava_pro(nn.Module):
    def __init__(self,
                 model_path='wisdomik/Quilt-Llava-v1.5-7b',
                 img_path="/home/quanyj/samw/datasets/TCGA/hnsc_svs/hnsc_jpgs",
                 patch_count=4,
                 prompt_base=""
                 ):
        super().__init__()
        #待实现
        self.num_patches = patch_count
        self.img_path = img_path

        self.qllava = QLlava(model_path,prompt_base=prompt_base)
        for param in self.qllava.parameters():
            param.requires_grad = False

    def crop_image_patches(self,folder_name,slide_id, num_patches):
        # 检查 num_patches 是否为正偶数
        if not isinstance(num_patches, int) or num_patches <= 0 or num_patches % 2 != 0:
            raise ValueError("num_patches must be a positive even integer.")
        image_path = os.path.join(folder_name, slide_id)
        # 加载图片
        image = Image.open(image_path)
        width, height = image.size
        # 计算每个 patch 的宽度和高度
        patch_width = width // (num_patches/2)
        patch_height = height // (num_patches/2)

        patches = []

        for i in range(int(num_patches/2)):
            for j in range(int(num_patches/2)):
                # 计算裁剪区域
                left = j * patch_width
                upper = i * patch_height
                right = left + patch_width
                lower = upper + patch_height

                # 裁剪图片

                patch = image.crop((left, upper, right, lower))
                patches.append(patch)
        patches.append(image)
        return patches

    def forward(self,data):

        slide_id_jpg,clinical_info = data
        ans_total = []  # 初始化空数组
        img_patchs = self.crop_image_patches(self.img_path, slide_id_jpg,self.num_patches)
        for index, img in enumerate(img_patchs):

            ans = self.qllava(img, clinical_info)

            
            ans_total.append(ans)  # 将每个ans添加到ans_total
        ans_total = [s.replace('</s>', '').replace('\n', '') for s in ans_total]

        return ans_total
    

            


if __name__ == '__main__':
    model = CPKSModel()
    image_folder = '/home/ubuntu/disk1/wys/SurvPath/imgs/blca_jpgs'
    csv_file = '/home/ubuntu/disk1/wys/SurvPath/datasets_csv/metadata/tcga_blca.csv'
    # dataset = PatchDataset(image_folder, csv_file)
    # dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    df = pd.read_csv(csv_file)
    image_path = os.path.join(image_folder, df.iloc[0]['slide_id']).replace("svs","jpg")
    print(image_path)
    image = Image.open(image_path).convert("RGB")
    age = df.iloc[0]['age']    
        # 切割成四个patch
    width, height = image.size
    half_width = width // 2
    half_height = height // 2

    patches = [
            image.crop((0, 0, half_width, half_height)),  # 左上角
            image.crop((half_width, 0, width, half_height)),  # 右上角
            image.crop((0, half_height, half_width, height)),  # 左下角
            image.crop((half_width, half_height, width, height)),  # 右下角
            image
        ]
    age = df.iloc[0]['age']
    gender = "Female" if df.iloc[0]['is_female'] == 1 else "Male"
        
        # 创建临床信息字典
    clinical_info = {
            "age": age,
            "gender": gender
        }
    description = f"This Bladder Urothelial Carcinoma patient is {age} years old and is {gender}."

    data = (patches,description)
    x = model(data)
    print(x)