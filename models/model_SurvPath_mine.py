
import torch
import numpy as np 
# from x_transformers import CrossAttender
import torch.nn as nn
# from torch_geometric.nn import SAGEConv
# from torch_geometric.utils import dense_to_sparse

import torch.nn.functional as F
import ipdb

from einops import reduce

# from x_transformers import Encoder
from torch.nn import ReLU

from models.layers.cross_attention import FeedForward, MMAttentionLayer
import pdb
from models.model_SurvPath import SurvPath
import math
import pandas as pd
from models.ourmodel import CPKSModel,CPKSModel1

def exists(val):
    return val is not None


def SNN_Block(dim1, dim2, dropout=0.25):
    r"""
    Multilayer Reception Block w/ Self-Normalization (Linear + ELU + Alpha Dropout)

    args:
        dim1 (int): Dimension of input features
        dim2 (int): Dimension of output features
        dropout (float): Dropout rate
    """

    return nn.Sequential(
            nn.Linear(dim1, dim2),
            nn.ELU(),
            nn.AlphaDropout(p=dropout, inplace=False))

#两阶段版
class SurvPath_with_Plugin(nn.Module):
    def __init__(self,
                omic_sizes=[100, 200, 300, 400, 500, 600],
                wsi_embedding_dim=768,
                dropout=0.1,
                num_classes=4,
                wsi_projection_dim=256,
                omic_names = [],
                study = 'stad',
                model_p= None,
                fusion= '',
                pk_path= '',
                global_act = True,  
                batch_size = 1    
                 ):
        super(SurvPath_with_Plugin, self).__init__()
        #下面这部分只影响到SurvPath原本的部分
        self.study = study
        self.model_p = model_p
        self.pk_path = pk_path
        self.model_p = model_p
        self.fusion = fusion
        for param in self.model_p.parameters():
            param.requires_grad = False
        self.cpks_plugin = CPKSModel(out_dim=model_p.out_dim_p,global_act = global_act,batch_size = batch_size)
        self.num_pathways = len(omic_sizes)

        if fusion == "concat_linear":
            self.linear = nn.Sequential(*[nn.Linear(model_p.out_dim_p*2, model_p.out_dim_p*4), nn.ReLU(), nn.Linear(model_p.out_dim_p*4, model_p.out_dim_p), nn.ReLU()])

        self.logits = nn.Sequential(
                nn.Linear(model_p.out_dim_p, int(model_p.out_dim_p/4)),
                nn.ReLU(),
                nn.Linear(int(model_p.out_dim_p/4), num_classes)
            )

    
    def forward(self, **kwargs):
        #process plugin
        self.cuda()

        
        #---> project omics data to projection_dim/2
        data = self.model_p.forward_p(**kwargs).unsqueeze(0) #[B, n]

        knowledge_df = pd.read_csv(self.pk_path)
        #process plugin
        # 将 slide_id 转换为 JPG 格式
        slide_ids_jpg = [slide_id.replace(".svs", ".jpg") for slide_id in kwargs['slide_id']]

        # 从 DataFrame 中筛选出所有符合条件的行
        matching_rows = knowledge_df[knowledge_df['Label'].isin(slide_ids_jpg)]

        # 按照 slide_ids_jpg 的顺序排列行
        matching_rows = matching_rows.set_index('Label').loc[slide_ids_jpg]

        # 提取 Ans_1 到 Ans_5 列的值并转换为 numpy 数组
        ans_values = matching_rows[['Ans_1', 'Ans_2', 'Ans_3', 'Ans_4', 'Ans_5']].values


        cpks_inputs = ans_values
        knowledge_emb = self.cpks_plugin(cpks_inputs)
        if self.fusion == "add":
            # 方式1: 元素级相加
            result = knowledge_emb + data

        elif self.fusion == "multiply":
            # 方式2: 元素级相乘
            result = knowledge_emb * data
            
        elif self.fusion == "concat_linear":
            # 方式3: 拼接后使用线性层
            if data.dim() == 2:
                data = data.unsqueeze(0)
            concat = torch.cat((knowledge_emb, data), dim=2)  # 维度 (bs, 64)
            result = self.linear(concat)  # 通过线性层映射回 (bs, 32)
        # embedding = paths_postSA_embed #---> top bloc only
        # embedding = wsi_postSA_embed #---> bottom bloc only

        # embedding = torch.mean(mm_embed, dim=1)
        #---> get logits
        logits = self.logits(result).squeeze(0)

        return logits
    
class SurvPath_with_Plugin1(nn.Module):
    def __init__(self,
                omic_sizes=[100, 200, 300, 400, 500, 600],
                wsi_embedding_dim=768,
                dropout=0.1,
                num_classes=4,
                wsi_projection_dim=256,
                omic_names = [],
                 ):
        super(SurvPath_with_Plugin, self).__init__()
        #下面这部分只影响到SurvPath原本的部分
        self.survpath = SurvPath(omic_sizes,
            wsi_embedding_dim,
            dropout,
            num_classes,
            wsi_projection_dim,
            omic_names)
        for param in self.survpath.parameters():
            param.requires_grad = False
        self.cpks_plugin = CPKSModel1()
        self.logits = nn.Sequential(
                nn.Linear(128*3, int(128*3/4)),
                nn.ReLU(),
                nn.Linear(int(128*3/4), num_classes)
            )

    def forward(self, **kwargs):
        paths_postSA_embed,wsi_postSA_embed = self.survpath.forward_cut(**kwargs)
        knowledge_df = pd.read_csv('/home/ubuntu/disk1/wys/SurvPath/datasets_csv/prior_knowledge/stad/knowledge_p4.csv')
        
        #process plugin
        slide_id_jpg = kwargs['slide_id'].replace(".svs",".jpg")
        matching_row = knowledge_df[knowledge_df['Label'] == slide_id_jpg]
        ans_values = matching_row[['Ans_1', 'Ans_2', 'Ans_3', 'Ans_4', 'Ans_5']].values.flatten()

        cpks_inputs = ans_values
        knowledge_emb = self.cpks_plugin(cpks_inputs)
        embedding = torch.cat([paths_postSA_embed, wsi_postSA_embed,knowledge_emb], dim=1) #---> both branches
        # embedding = paths_postSA_embed #---> top bloc only
        # embedding = wsi_postSA_embed #---> bottom bloc only

        # embedding = torch.mean(mm_embed, dim=1)
        #---> get logits
        logits = self.logits(embedding)

        return logits





























































































#完整版
# class SurvPath_with_Plugin(nn.Module):
#     def __init__(self,
#                 omic_sizes=[100, 200, 300, 400, 500, 600],
#                 wsi_embedding_dim=768,
#                 dropout=0.1,
#                 num_classes=4,
#                 wsi_projection_dim=256,
#                 omic_names = [],
#                  ):
#         super(SurvPath_with_Plugin, self).__init__()
#         #下面这部分只影响到SurvPath原本的部分
#         self.survpath = SurvPath(omic_sizes,
#             wsi_embedding_dim,
#             dropout,
#             num_classes,
#             wsi_projection_dim,
#             omic_names)
#         for param in self.survpath.parameters():
#             param.requires_grad = False
#         self.cpks_plugin = CPKSModel()
#         self.logits = nn.Sequential(
#                 nn.Linear(128*3, int(128*3/4)),
#                 nn.ReLU(),
#                 nn.Linear(int(128*3/4), num_classes)
#             )

#     def forward(self, **kwargs):
#         paths_postSA_embed,wsi_postSA_embed = self.survpath.forward_cut(**kwargs)
        
#         #process plugin
#         slide_id_jpg = kwargs['slide_id'].replace("svs","jpg")
#         clinical_info = kwargs['clinical_data'][0]
#         clinical_prompt = f"The patient is at stage {clinical_info[0]}, grade {clinical_info[1]}, and has a cancer type of {clinical_info[2]}."
#         cpks_inputs = (slide_id_jpg,clinical_prompt)
#         knowledge_emb = self.cpks_plugin(cpks_inputs)
#         embedding = torch.cat([paths_postSA_embed, wsi_postSA_embed,knowledge_emb], dim=1) #---> both branches
#         # embedding = paths_postSA_embed #---> top bloc only
#         # embedding = wsi_postSA_embed #---> bottom bloc only

#         # embedding = torch.mean(mm_embed, dim=1)
#         #---> get logits
#         logits = self.logits(embedding)

#         return logits

#仅用插件特征预测    
# class Plugin(nn.Module):
#     def __init__(self,
#                 omic_sizes=[100, 200, 300, 400, 500, 600],
#                 wsi_embedding_dim=768,
#                 dropout=0.1,
#                 num_classes=4,
#                  ):
#         super(Plugin, self).__init__()
#         #下面这部分只影响到SurvPath原本的部分
#         self.cpks_plugin = CPKSModel()
#         self.logits = nn.Sequential(
#                 nn.Linear(128, int(128/4)),
#                 nn.ReLU(),
#                 nn.Linear(int(128/4), num_classes)
#             )

#     def forward(self, **kwargs):
        
#         #process plugin
#         slide_id_jpg = kwargs['slide_id'].replace("svs","jpg")
#         clinical_info = kwargs['clinical_data'][0]
#         clinical_prompt = f"The patient is at stage {clinical_info[0]}, grade {clinical_info[1]}, and has a cancer type of {clinical_info[2]}."
#         cpks_inputs = (slide_id_jpg,clinical_prompt)
#         knowledge_emb = self.cpks_plugin(cpks_inputs)
#         # embedding = paths_postSA_embed #---> top bloc only
#         # embedding = wsi_postSA_embed #---> bottom bloc only

#         # embedding = torch.mean(mm_embed, dim=1)
#         #---> get logits
#         logits = self.logits(knowledge_emb)

#         return logits

#给每个通路加注意力
# class SurvPath_mine(nn.Module):
#     def __init__(
#         self, 
#         omic_sizes=[100, 200, 300, 400, 500, 600],
#         wsi_embedding_dim=768,
#         dropout=0.1,
#         num_classes=4,
#         wsi_projection_dim=256,
#         omic_names = [],
#         batch_first=True
#         ):
#         super(SurvPath_mine, self).__init__()

#         #---> general props
#         self.num_pathways = len(omic_sizes)
#         self.dropout = dropout

#         #---> omics preprocessing for captum
#         if omic_names != []:
#             self.omic_names = omic_names
#             all_gene_names = []
#             for group in omic_names:
#                 all_gene_names.append(group)
#             all_gene_names = np.asarray(all_gene_names)
#             all_gene_names = np.concatenate(all_gene_names)
#             all_gene_names = np.unique(all_gene_names)
#             all_gene_names = list(all_gene_names)
#             self.all_gene_names = all_gene_names
#         self.gnn_omic = nn.Sequential(
#             # 第一层: 从256维升到512维
#             SAGEConv(wsi_projection_dim, wsi_projection_dim*2),
#             nn.ReLU(),  # 激活函数
#             # 第二层: 从512维升到768维
#             SAGEConv(wsi_projection_dim*2, wsi_embedding_dim),
#             nn.ReLU()   # 激活函数
#         )
#         # self.fc = nn.Linear(input_dim=268, output_dim=256)  # 将自注意力的结果映射到 256 维
#         #---> wsi props
#         self.wsi_embedding_dim = wsi_embedding_dim 
#         self.wsi_projection_dim = wsi_projection_dim

#         self.wsi_projection_net = nn.Sequential(
#             nn.Linear(self.wsi_embedding_dim, self.wsi_projection_dim),
#         )

#         #---> omics props
#         self.init_per_path_model(omic_sizes)

#         #---> cross attention props
#         self.identity = nn.Identity() # use this layer to calculate ig
#         self.cross_attender = MMAttentionLayer(
#             dim=self.wsi_projection_dim,
#             dim_head=self.wsi_projection_dim // 2,
#             heads=1,
#             residual=False,
#             dropout=0.1,
#             num_pathways = self.num_pathways
#         )

#         #---> logits props 
#         self.num_classes = num_classes
#         self.feed_forward = FeedForward(self.wsi_projection_dim // 2, dropout=dropout)
#         self.feed_forward2 = FeedForward(self.wsi_projection_dim, dropout=dropout)
#         self.layer_norm = nn.LayerNorm(self.wsi_projection_dim // 2)
#         self.layer_norm2 = nn.LayerNorm(self.wsi_projection_dim)
#         self.omic_attn=nn.MultiheadAttention(self.wsi_projection_dim, 8, dropout=dropout,batch_first=batch_first)
#         # when both top and bottom blocks 
#         self.to_logits = nn.Sequential(
#                 nn.Linear(self.wsi_projection_dim, int(self.wsi_projection_dim/4)),
#                 nn.ReLU(),
#                 nn.Linear(int(self.wsi_projection_dim/4), self.num_classes)
#             )
        
#     def init_per_path_model(self, omic_sizes):
#         hidden = [256, 256]
#         sig_networks = []
#         for input_dim in omic_sizes:
#             fc_omic = [SNN_Block(dim1=input_dim, dim2=hidden[0])]
#             for i, _ in enumerate(hidden[1:]):
#                 fc_omic.append(SNN_Block(dim1=hidden[i], dim2=hidden[i+1], dropout=0.25))
#             sig_networks.append(nn.Sequential(*fc_omic))
#         self.sig_networks = nn.ModuleList(sig_networks)    
    
#     def pad(self,feature):
#         padding = self.wsi_projection_dim - feature.shape[0]
#         processed_feature = F.pad(feature, (0, padding)) 
#         return processed_feature
    

#     def forward(self, **kwargs):

#         wsi = kwargs['x_path']
#         x_omic = [kwargs['x_omic%d' % i] for i in range(1,self.num_pathways+1)]
#         mask = None
#         return_attn = kwargs["return_attn"]
#         features = [self.pad(arr) for arr in x_omic]  # 将输入列表中的数组转为 Tensor
#         padded_features = torch.stack(features).unsqueeze(0)
#         # 2. 通过自注意力机制处理
#         omic_attn_output,_ = self.omic_attn(padded_features,padded_features,padded_features)  # 输出形状为 (331, max_len, input_dim)
#         #---> get pathway embeddings 
#         h_omic = [self.sig_networks[idx].forward(sig_feat.float()) for idx, sig_feat in enumerate(x_omic)] ### each omic signature goes through it's own FC layer
#         h_omic_bag = torch.stack(h_omic).unsqueeze(0) ### omic embeddings are stacked (to be used in co-attention)
#         h_omic_bag =self.feed_forward2(omic_attn_output+h_omic_bag)
#         h_omic_bag = self.layer_norm2(h_omic_bag)
#         # num_nodes = h_omic_bag.size(0)
#         # edge_index, _ = dense_to_sparse((torch.ones(num_nodes, num_nodes) - torch.eye(num_nodes)).to("cuda:0"))
#         # for layer in self.gnn_omic:
#         #     if isinstance(layer, SAGEConv):
#         #         h_omic_bag = layer(h_omic_bag, edge_index)  # 传入特征和边连接
#         #     else:
#         #         h_omic_bag = layer(h_omic_bag)  # 传入特征，激活函数只需要特征

#         #---> project wsi to smaller dimension (same as pathway dimension)
#         wsi_embed = self.wsi_projection_net(wsi)

#         tokens = torch.cat([h_omic_bag, wsi_embed], dim=1)
#         tokens = self.identity(tokens)
        
#         if return_attn:
#             mm_embed, attn_pathways, cross_attn_pathways, cross_attn_histology = self.cross_attender(x=tokens, mask=mask if mask is not None else None, return_attention=True)
#         else:
#             mm_embed = self.cross_attender(x=tokens, mask=mask if mask is not None else None, return_attention=False)

#         #---> feedforward and layer norm 
#         mm_embed = self.feed_forward(mm_embed)
#         mm_embed = self.layer_norm(mm_embed)
        
#         #---> aggregate 
#         # modality specific mean 
#         paths_postSA_embed = mm_embed[:, :self.num_pathways, :]
#         paths_postSA_embed = torch.mean(paths_postSA_embed, dim=1)

#         wsi_postSA_embed = mm_embed[:, self.num_pathways:, :]
#         wsi_postSA_embed = torch.mean(wsi_postSA_embed, dim=1)

#         # when both top and bottom block
#         embedding = torch.cat([paths_postSA_embed, wsi_postSA_embed], dim=1) #---> both branches
#         # embedding = paths_postSA_embed #---> top bloc only
#         # embedding = wsi_postSA_embed #---> bottom bloc only

#         # embedding = torch.mean(mm_embed, dim=1)
#         #---> get logits
#         logits = self.to_logits(embedding)

#         if return_attn:
#             return logits, attn_pathways, cross_attn_pathways, cross_attn_histology
#         else:
#             return logits
        
#     def captum(self, omics_0 ,omics_1 ,omics_2 ,omics_3 ,omics_4 ,omics_5 ,omics_6 ,omics_7 ,omics_8 ,omics_9 ,omics_10 ,omics_11 ,omics_12 ,omics_13 ,omics_14 ,omics_15 ,omics_16 ,omics_17 ,omics_18 ,omics_19 ,omics_20 ,omics_21 ,omics_22 ,omics_23 ,omics_24 ,omics_25 ,omics_26 ,omics_27 ,omics_28 ,omics_29 ,omics_30 ,omics_31 ,omics_32 ,omics_33 ,omics_34 ,omics_35 ,omics_36 ,omics_37 ,omics_38 ,omics_39 ,omics_40 ,omics_41 ,omics_42 ,omics_43 ,omics_44 ,omics_45 ,omics_46 ,omics_47 ,omics_48 ,omics_49 ,omics_50 ,omics_51 ,omics_52 ,omics_53 ,omics_54 ,omics_55 ,omics_56 ,omics_57 ,omics_58 ,omics_59 ,omics_60 ,omics_61 ,omics_62 ,omics_63 ,omics_64 ,omics_65 ,omics_66 ,omics_67 ,omics_68 ,omics_69 ,omics_70 ,omics_71 ,omics_72 ,omics_73 ,omics_74 ,omics_75 ,omics_76 ,omics_77 ,omics_78 ,omics_79 ,omics_80 ,omics_81 ,omics_82 ,omics_83 ,omics_84 ,omics_85 ,omics_86 ,omics_87 ,omics_88 ,omics_89 ,omics_90 ,omics_91 ,omics_92 ,omics_93 ,omics_94 ,omics_95 ,omics_96 ,omics_97 ,omics_98 ,omics_99 ,omics_100 ,omics_101 ,omics_102 ,omics_103 ,omics_104 ,omics_105 ,omics_106 ,omics_107 ,omics_108 ,omics_109 ,omics_110 ,omics_111 ,omics_112 ,omics_113 ,omics_114 ,omics_115 ,omics_116 ,omics_117 ,omics_118 ,omics_119 ,omics_120 ,omics_121 ,omics_122 ,omics_123 ,omics_124 ,omics_125 ,omics_126 ,omics_127 ,omics_128 ,omics_129 ,omics_130 ,omics_131 ,omics_132 ,omics_133 ,omics_134 ,omics_135 ,omics_136 ,omics_137 ,omics_138 ,omics_139 ,omics_140 ,omics_141 ,omics_142 ,omics_143 ,omics_144 ,omics_145 ,omics_146 ,omics_147 ,omics_148 ,omics_149 ,omics_150 ,omics_151 ,omics_152 ,omics_153 ,omics_154 ,omics_155 ,omics_156 ,omics_157 ,omics_158 ,omics_159 ,omics_160 ,omics_161 ,omics_162 ,omics_163 ,omics_164 ,omics_165 ,omics_166 ,omics_167 ,omics_168 ,omics_169 ,omics_170 ,omics_171 ,omics_172 ,omics_173 ,omics_174 ,omics_175 ,omics_176 ,omics_177 ,omics_178 ,omics_179 ,omics_180 ,omics_181 ,omics_182 ,omics_183 ,omics_184 ,omics_185 ,omics_186 ,omics_187 ,omics_188 ,omics_189 ,omics_190 ,omics_191 ,omics_192 ,omics_193 ,omics_194 ,omics_195 ,omics_196 ,omics_197 ,omics_198 ,omics_199 ,omics_200 ,omics_201 ,omics_202 ,omics_203 ,omics_204 ,omics_205 ,omics_206 ,omics_207 ,omics_208 ,omics_209 ,omics_210 ,omics_211 ,omics_212 ,omics_213 ,omics_214 ,omics_215 ,omics_216 ,omics_217 ,omics_218 ,omics_219 ,omics_220 ,omics_221 ,omics_222 ,omics_223 ,omics_224 ,omics_225 ,omics_226 ,omics_227 ,omics_228 ,omics_229 ,omics_230 ,omics_231 ,omics_232 ,omics_233 ,omics_234 ,omics_235 ,omics_236 ,omics_237 ,omics_238 ,omics_239 ,omics_240 ,omics_241 ,omics_242 ,omics_243 ,omics_244 ,omics_245 ,omics_246 ,omics_247 ,omics_248 ,omics_249 ,omics_250 ,omics_251 ,omics_252 ,omics_253 ,omics_254 ,omics_255 ,omics_256 ,omics_257 ,omics_258 ,omics_259 ,omics_260 ,omics_261 ,omics_262 ,omics_263 ,omics_264 ,omics_265 ,omics_266 ,omics_267 ,omics_268 ,omics_269 ,omics_270 ,omics_271 ,omics_272 ,omics_273 ,omics_274 ,omics_275 ,omics_276 ,omics_277 ,omics_278 ,omics_279 ,omics_280 ,omics_281 ,omics_282 ,omics_283 ,omics_284 ,omics_285 ,omics_286 ,omics_287 ,omics_288 ,omics_289 ,omics_290 ,omics_291 ,omics_292 ,omics_293 ,omics_294 ,omics_295 ,omics_296 ,omics_297 ,omics_298 ,omics_299 ,omics_300 ,omics_301 ,omics_302 ,omics_303 ,omics_304 ,omics_305 ,omics_306 ,omics_307 ,omics_308 ,omics_309 ,omics_310 ,omics_311 ,omics_312 ,omics_313 ,omics_314 ,omics_315 ,omics_316 ,omics_317 ,omics_318 ,omics_319 ,omics_320 ,omics_321 ,omics_322 ,omics_323 ,omics_324 ,omics_325 ,omics_326 ,omics_327 ,omics_328 ,omics_329 ,omics_330, wsi):
        
#         #---> unpack inputs
#         mask = None
#         return_attn = False
        
#         omic_list = [omics_0 ,omics_1 ,omics_2 ,omics_3 ,omics_4 ,omics_5 ,omics_6 ,omics_7 ,omics_8 ,omics_9 ,omics_10 ,omics_11 ,omics_12 ,omics_13 ,omics_14 ,omics_15 ,omics_16 ,omics_17 ,omics_18 ,omics_19 ,omics_20 ,omics_21 ,omics_22 ,omics_23 ,omics_24 ,omics_25 ,omics_26 ,omics_27 ,omics_28 ,omics_29 ,omics_30 ,omics_31 ,omics_32 ,omics_33 ,omics_34 ,omics_35 ,omics_36 ,omics_37 ,omics_38 ,omics_39 ,omics_40 ,omics_41 ,omics_42 ,omics_43 ,omics_44 ,omics_45 ,omics_46 ,omics_47 ,omics_48 ,omics_49 ,omics_50 ,omics_51 ,omics_52 ,omics_53 ,omics_54 ,omics_55 ,omics_56 ,omics_57 ,omics_58 ,omics_59 ,omics_60 ,omics_61 ,omics_62 ,omics_63 ,omics_64 ,omics_65 ,omics_66 ,omics_67 ,omics_68 ,omics_69 ,omics_70 ,omics_71 ,omics_72 ,omics_73 ,omics_74 ,omics_75 ,omics_76 ,omics_77 ,omics_78 ,omics_79 ,omics_80 ,omics_81 ,omics_82 ,omics_83 ,omics_84 ,omics_85 ,omics_86 ,omics_87 ,omics_88 ,omics_89 ,omics_90 ,omics_91 ,omics_92 ,omics_93 ,omics_94 ,omics_95 ,omics_96 ,omics_97 ,omics_98 ,omics_99 ,omics_100 ,omics_101 ,omics_102 ,omics_103 ,omics_104 ,omics_105 ,omics_106 ,omics_107 ,omics_108 ,omics_109 ,omics_110 ,omics_111 ,omics_112 ,omics_113 ,omics_114 ,omics_115 ,omics_116 ,omics_117 ,omics_118 ,omics_119 ,omics_120 ,omics_121 ,omics_122 ,omics_123 ,omics_124 ,omics_125 ,omics_126 ,omics_127 ,omics_128 ,omics_129 ,omics_130 ,omics_131 ,omics_132 ,omics_133 ,omics_134 ,omics_135 ,omics_136 ,omics_137 ,omics_138 ,omics_139 ,omics_140 ,omics_141 ,omics_142 ,omics_143 ,omics_144 ,omics_145 ,omics_146 ,omics_147 ,omics_148 ,omics_149 ,omics_150 ,omics_151 ,omics_152 ,omics_153 ,omics_154 ,omics_155 ,omics_156 ,omics_157 ,omics_158 ,omics_159 ,omics_160 ,omics_161 ,omics_162 ,omics_163 ,omics_164 ,omics_165 ,omics_166 ,omics_167 ,omics_168 ,omics_169 ,omics_170 ,omics_171 ,omics_172 ,omics_173 ,omics_174 ,omics_175 ,omics_176 ,omics_177 ,omics_178 ,omics_179 ,omics_180 ,omics_181 ,omics_182 ,omics_183 ,omics_184 ,omics_185 ,omics_186 ,omics_187 ,omics_188 ,omics_189 ,omics_190 ,omics_191 ,omics_192 ,omics_193 ,omics_194 ,omics_195 ,omics_196 ,omics_197 ,omics_198 ,omics_199 ,omics_200 ,omics_201 ,omics_202 ,omics_203 ,omics_204 ,omics_205 ,omics_206 ,omics_207 ,omics_208 ,omics_209 ,omics_210 ,omics_211 ,omics_212 ,omics_213 ,omics_214 ,omics_215 ,omics_216 ,omics_217 ,omics_218 ,omics_219 ,omics_220 ,omics_221 ,omics_222 ,omics_223 ,omics_224 ,omics_225 ,omics_226 ,omics_227 ,omics_228 ,omics_229 ,omics_230 ,omics_231 ,omics_232 ,omics_233 ,omics_234 ,omics_235 ,omics_236 ,omics_237 ,omics_238 ,omics_239 ,omics_240 ,omics_241 ,omics_242 ,omics_243 ,omics_244 ,omics_245 ,omics_246 ,omics_247 ,omics_248 ,omics_249 ,omics_250 ,omics_251 ,omics_252 ,omics_253 ,omics_254 ,omics_255 ,omics_256 ,omics_257 ,omics_258 ,omics_259 ,omics_260 ,omics_261 ,omics_262 ,omics_263 ,omics_264 ,omics_265 ,omics_266 ,omics_267 ,omics_268 ,omics_269 ,omics_270 ,omics_271 ,omics_272 ,omics_273 ,omics_274 ,omics_275 ,omics_276 ,omics_277 ,omics_278 ,omics_279 ,omics_280 ,omics_281 ,omics_282 ,omics_283 ,omics_284 ,omics_285 ,omics_286 ,omics_287 ,omics_288 ,omics_289 ,omics_290 ,omics_291 ,omics_292 ,omics_293 ,omics_294 ,omics_295 ,omics_296 ,omics_297 ,omics_298 ,omics_299 ,omics_300 ,omics_301 ,omics_302 ,omics_303 ,omics_304 ,omics_305 ,omics_306 ,omics_307 ,omics_308 ,omics_309 ,omics_310 ,omics_311 ,omics_312 ,omics_313 ,omics_314 ,omics_315 ,omics_316 ,omics_317 ,omics_318 ,omics_319 ,omics_320 ,omics_321 ,omics_322 ,omics_323 ,omics_324 ,omics_325 ,omics_326 ,omics_327 ,omics_328 ,omics_329 ,omics_330]

#         #---> get pathway embeddings 
#         h_omic = [self.sig_networks[idx].forward(sig_feat) for idx, sig_feat in enumerate(omic_list)] ### each omic signature goes through it's own FC layer
#         h_omic_bag = torch.stack(h_omic, dim=1)  #.unsqueeze(0) ### omic embeddings are stacked (to be used in co-attention)
        
#         #---> project wsi to smaller dimension (same as pathway dimension)
#         wsi_embed = self.wsi_projection_net(wsi)

#         tokens = torch.cat([h_omic_bag, wsi_embed], dim=1)
#         tokens = self.identity(tokens)

#         if return_attn:
#             mm_embed, attn_pathways, cross_attn_pathways, cross_attn_histology = self.cross_attender(x=tokens, mask=mask if mask is not None else None, return_attention=True)
#         else:
#             mm_embed = self.cross_attender(x=tokens, mask=mask if mask is not None else None, return_attention=False)

#         #---> feedforward and layer norm 
#         mm_embed = self.feed_forward(mm_embed)
#         mm_embed = self.layer_norm(mm_embed)
        
#         #---> aggregate 
#         # modality specific mean 
#         paths_postSA_embed = mm_embed[:, :self.num_pathways, :]
#         paths_postSA_embed = torch.mean(paths_postSA_embed, dim=1)

#         wsi_postSA_embed = mm_embed[:, self.num_pathways:, :]
#         wsi_postSA_embed = torch.mean(wsi_postSA_embed, dim=1)
#         embedding = torch.cat([paths_postSA_embed, wsi_postSA_embed], dim=1)

#         #---> get logits
#         logits = self.to_logits(embedding)

#         hazards = torch.sigmoid(logits)
#         survival = torch.cumprod(1 - hazards, dim=1)
#         risk = -torch.sum(survival, dim=1)

#         if return_attn:
#             return risk, attn_pathways, cross_attn_pathways, cross_attn_histology
#         else:
#             return risk