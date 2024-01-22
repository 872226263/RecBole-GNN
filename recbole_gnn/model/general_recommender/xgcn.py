import torch
import torch.nn as nn
from recbole.model.abstract_recommender import GeneralRecommender
from recbole.model.loss import BPRLoss
from recbole.utils import InputType
from recbole_gnn.model.layers import LightGCNConv

class xGCN(GeneralGraphRecommender):
    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset):
        super(xGCN, self).__init__(config, dataset)

        # load parameters info
        self.latent_dim = config['embedding_size']  # int type:the embedding size of lightGCN
        self.n_layers = config['n_layers']  # int type:the layer num of lightGCN
        self.reg_weight = config['reg_weight']  # float32 type: the weight decay for l2 normalization
        self.require_pow = config['require_pow']  # bool type: whether to require pow when regularization

        # define layers and loss
        self.user_embedding = nn.Embedding(num_embeddings=self.n_users, embedding_dim=self.latent_dim)
        self.item_embedding = nn.Embedding(num_embeddings=self.n_items, embedding_dim=self.latent_dim)
        self.gcn_conv = LightGCNConv(dim=self.latent_dim)
        self.mf_loss = BPRLoss()
        self.reg_loss = EmbLoss()

        # Custom RefNet initialization
        self.create_refnet()

        # Optimizer
        self.optimizer = torch.optim.Adam([
            {'params': self.refnet.parameters(), 'lr': config['dnn_lr']},
        ])

    def create_refnet(self):
        dnn_arch = self.config['dnn_arch']
        scale_net_arch = self.config['scale_net_arch'] if self.config['use_scale_net'] else None
        self.refnet = RefNet(dnn_arch, scale_net_arch).to(self.forward_device)

    def forward(self, user_indices, item_indices):
        # Assuming user_indices and item_indices are provided by RecBole
        user_emb = self.get_refnet_output_emb(user_indices)
        item_emb = self.get_refnet_output_emb(item_indices)
        scores = dot_product(user_emb, item_emb)
        return scores

    def calculate_loss(self, interaction):
        user = interaction[self.USER_ID]
        pos_item = interaction[self.ITEM_ID]
        neg_item = interaction[self.NEG_ITEM_ID]

        user_emb = self.get_refnet_output_emb(user)
        pos_item_emb = self.get_refnet_output_emb(pos_item)
        neg_item_emb = self.get_refnet_output_emb(neg_item)

        pos_score = dot_product(user_emb, pos_item_emb)
        neg_score = dot_product(user_emb, neg_item_emb)

        if self.loss_type == 'bpr':
            loss = bpr_loss(pos_score, neg_score)
        elif self.loss_type == 'bce':
            loss = bce_loss(pos_score, neg_score)

        if self.reg_weight > 0:
            L2_reg_loss = 1/2 * (1 / len(user)) * (
                (user_emb**2).sum() + (pos_item_emb**2).sum() + (neg_item_emb**2).sum()
            )
            loss += self.reg_weight * L2_reg_loss

        return loss

    def get_refnet_output_emb(self, nids):
        emb = self.refnet(self.emb_table[nids].to(self.forward_device))
        return emb

    # Implement other necessary methods from xGCN as needed
