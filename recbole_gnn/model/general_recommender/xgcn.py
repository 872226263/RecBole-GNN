import torch
import torch.nn as nn
from recbole.model.abstract_recommender import GeneralRecommender
from recbole.model.loss import BPRLoss
from recbole.utils import InputType

# Import necessary modules from XGCN
from XGCN.model.module.propagation import LightGCN_Propagation
from XGCN.model.module import init_emb_table, dot_product, bpr_loss, bce_loss
from .module import RefNet

class xGCN(GeneralRecommender):
    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset):
        super(xGCN, self).__init__(config, dataset)

        # Configuration and dataset handling
        self.emb_table_device = config['emb_table_device']
        self.forward_device = config['forward_device']
        self.out_emb_table_device = config['out_emb_table_device']
        self.num_nodes = dataset.num_nodes  # Assuming num_nodes is a dataset attribute

        # xGCN specific initializations
        self.propagation = LightGCN_Propagation(config, dataset)
        self.emb_table = init_emb_table(config, self.num_nodes, return_tensor=True)
        self.create_refnet()
        self.out_emb_table = torch.empty(self.emb_table.shape, dtype=torch.float32).to(self.out_emb_table_device)

        # DataLoader for nodes
        self.node_dl = torch.utils.data.DataLoader(torch.arange(self.num_nodes), batch_size=4096)

        # Optimizer
        self.optimizer = torch.optim.Adam([
            {'params': self.refnet.parameters(), 'lr': config['dnn_lr']},
        ])

        # Additional attributes
        self.epoch_last_prop = 0
        self.total_prop_times = 0

        # Loss function
        self.loss_type = config['loss_type']
        self.reg_weight = config['L2_reg_weight']
        if self.loss_type not in ['bpr', 'bce']:
            raise ValueError("Invalid loss type. Choose 'bpr' or 'bce'.")

        # Propagation
        if not config['from_pretrained']:
            self.do_propagation()

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
        user = interaction['user_id']
        pos_item = interaction['pos_item_id']
        neg_item = interaction['neg_item_id']

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
