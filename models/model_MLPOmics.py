import torch.nn as nn
from torch.nn import ReLU, ELU
import torch
from models.ourmodel import CPKSModel
from models.model_utils import BilinearFusion
import pandas as pd
"""

Implement a MLP to handle tabular omics data 

"""

class MLPOmics(nn.Module):
    def __init__(
        self, 
        input_dim,
        n_classes=4, 
        projection_dim = 512, 
        dropout = 0.1, 
        ):
        super(MLPOmics, self).__init__()
        
        # self
        self.projection_dim = projection_dim

        self.net = nn.Sequential(
            nn.Linear(input_dim, projection_dim//2), ReLU(), nn.Dropout(dropout),
            nn.Linear(projection_dim//2, projection_dim//2), ReLU(), nn.Dropout(dropout)
        ) 
        
        self.to_logits = nn.Sequential(
                nn.Linear(projection_dim//2, n_classes)
            )

    def forward(self, **kwargs):
        self.cuda()

        #---> unpack
        data_omics = kwargs["data_omics"].float().cuda().squeeze()
        
        #---> project omics data to projection_dim/2
        data = self.net(data_omics) #[B, n]

        #---->predict
        logits = self.to_logits(data) #[B, n_classes]
        return logits
    
    def captum(self, omics):

        self.cuda()

        #---> unpack
        data_omics = omics.float().cuda().squeeze()
        
        #---> project omics data to projection_dim/2
        data = self.net(data_omics) #[B, n]

        #---->predict
        logits = self.to_logits(data) #[B, n_classes]

        #---> get risk 
        hazards = torch.sigmoid(logits)
        survival = torch.cumprod(1 - hazards, dim=1)
        risk = -torch.sum(survival, dim=1)

        #---> return risk 
        return risk
    
class MLPOmicswithP(nn.Module):
    def __init__(
        self, 
        input_dim,
        n_classes=4, 
        projection_dim = 512, 
        dropout = 0.1, 
        study = 'stad',
        fusion = 'add',
        global_act = True
        ):
        super(MLPOmicswithP, self).__init__()
        
        # self
        
        self.projection_dim = projection_dim
        self.study = study
        self.fusion = fusion
        self.net = nn.Sequential(
            nn.Linear(input_dim, projection_dim//2), ReLU(), nn.Dropout(dropout),
            nn.Linear(projection_dim//2, projection_dim//2), ReLU(), nn.Dropout(dropout)
        ) 
        for param in self.net.parameters():
            param.requires_grad = False

        self.to_logits = nn.Sequential(
                nn.Linear(projection_dim//2, n_classes)
            )
        
        self.cpks_plugin = CPKSModel(out_dim=projection_dim//2,global_act = global_act)
        self.linear = nn.Sequential(*[nn.Linear(projection_dim, projection_dim*4), nn.ReLU(), nn.Linear(projection_dim*4, projection_dim//2), nn.ReLU()])
        self.bilinear = BilinearFusion(dim1=256, dim2=256, scale_dim1=8, scale_dim2=8, mmhid=256)


    def forward(self, **kwargs):
        self.cuda()

        #---> unpack
        data_omics = kwargs["data_omics"].float().cuda().squeeze()
        
        #---> project omics data to projection_dim/2
        data = self.net(data_omics).unsqueeze(0) #[B, n]

        knowledge_df = pd.read_csv(f'/root/autodl-tmp/survival_bag/data/prior_knowledge/{self.study}/knowledge_p4.csv')
        
        #process plugin
        slide_id_jpg = kwargs['slide_id'].replace(".svs",".jpg")
        matching_row = knowledge_df[knowledge_df['Label'] == slide_id_jpg]
        ans_values = matching_row[['Ans_1', 'Ans_2', 'Ans_3', 'Ans_4', 'Ans_5']].values.flatten()

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
            concat = torch.cat((knowledge_emb, data), dim=1)  # 维度 (1, 64)
            result = self.linear(concat)  # 通过线性层映射回 (1, 32)

            
        else:
            raise ValueError("Unsupported fusion method. Choose from 'add', 'multiply', or 'concat_linear'.")

        #---->predict
        logits = self.to_logits(result) #[B, n_classes]
        return logits
    



