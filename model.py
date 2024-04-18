import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import dgl.function as fn
import numpy as np


class DQGCN(nn.Module):
    def __init__(self, num_ents, num_rels,args):
        super(DQGCN, self).__init__()

        self.num_rels = num_rels
        self.num_ents = num_ents
        self.h_dim = args.n_hidden
        self.gpu = args.gpu
        self.alpha = args.alpha
        self.pi = 3.14159265358979323846

        self.emb_rel = torch.nn.Parameter(torch.Tensor(num_rels*2, self.h_dim), requires_grad=True).float()
        self.static_emb = torch.nn.Parameter(torch.Tensor(num_ents, self.h_dim), requires_grad=True).float()
        self.alpha_t = torch.nn.Parameter(torch.Tensor(num_ents, self.h_dim), requires_grad=True).float()
        self.beta_t = torch.nn.Parameter(torch.Tensor(num_ents, self.h_dim), requires_grad=True).float()
        self.temporal_w = torch.nn.Parameter(torch.Tensor(self.h_dim*2, self.h_dim), requires_grad=True).float()
        self.time_gate = nn.Linear(self.h_dim,self.h_dim)

        torch.nn.init.xavier_normal_(self.emb_rel)
        torch.nn.init.normal_(self.static_emb)
        torch.nn.init.normal_(self.alpha_t)
        torch.nn.init.normal_(self.beta_t)
        torch.nn.init.normal_(self.temporal_w)

        self.loss_e = torch.nn.CrossEntropyLoss()
        
        self.rgcn = RGCNCell(self.h_dim,
                             self.h_dim,
                             num_rels * 2,
                             args.n_layers,
                             args.dropout,
                             self.emb_rel)

        self.decoder_ob = ConvTransE(num_ents, self.h_dim, args.dropout)

    def get_dynamic_emb(self,t):
        # return self.static_emb
        timevec = self.alpha * self.alpha_t*t + (1-self.alpha) * torch.sin(2 * self.pi * self.beta_t*t)
        attn = torch.cat([self.static_emb,timevec],1)
        return torch.mm(attn, self.temporal_w)

    def forward(self, g_list, err_mat, t):
        related_emb = torch.spmm(err_mat,self.emb_rel)
        self.inputs = [F.normalize(self.get_dynamic_emb(t))]
        for i, g in enumerate(g_list):
            g = g.to(self.gpu)
            cur_output = F.normalize(self.rgcn.forward(g, self.inputs[-1]))
            self.inputs.append(self.get_composed(cur_output, related_emb))
            # self.inputs.append(cur_output)
        return self.inputs[-1], self.emb_rel


    def get_composed(self,cur_output, related_emb):
        self.time_weights = []
        for i in range(len(self.inputs)):
            self.time_weights.append(self.time_gate(self.inputs[i]+related_emb))
        self.time_weights.append(torch.zeros(self.num_ents,self.h_dim).cuda())
        self.time_weights = torch.stack(self.time_weights,0)
        self.time_weights = torch.softmax(self.time_weights,0)
        output = cur_output*self.time_weights[-1]
        for i in range(len(self.inputs)):
            output += self.time_weights[i]*self.inputs[i]
        return F.normalize(output)


    def predict(self, test_graph, tlist, test_triplets1, test_triplets2, err_mat1, err_mat2):
        with torch.no_grad():
            evolve_embs, r_emb = self.forward(test_graph, err_mat1, tlist[0])
            score1 = self.decoder_ob.forward(evolve_embs, r_emb, test_triplets1)

            evolve_embs, r_emb = self.forward(test_graph, err_mat2, tlist[0])
            score2 = self.decoder_ob.forward(evolve_embs, r_emb, test_triplets2)

            score = torch.cat([score1,score2])
            score = torch.softmax(score, dim=1)
            return score
            

    def get_loss(self, glist, tlist, triples, err_mat):
        evolve_embs, r_emb = self.forward(glist, err_mat, tlist[0])
        scores_ob = self.decoder_ob.forward(evolve_embs, r_emb, triples)
        loss_ent = self.loss_e(scores_ob, triples[:, 2])
        return loss_ent



class RGCNCell(nn.Module):
    def __init__(self, h_dim, out_dim, num_rels, 
                 num_hidden_layers=1, dropout=0,  rel_emb=None):
        super(RGCNCell, self).__init__()
        self.h_dim = h_dim
        self.out_dim = out_dim
        self.num_rels = num_rels
        self.num_hidden_layers = num_hidden_layers
        self.dropout = dropout
        self.rel_emb = rel_emb

        self.layers = nn.ModuleList()
        for idx in range(self.num_hidden_layers):
            h2h = UnionRGCNLayer(self.h_dim, self.out_dim, self.num_rels,
                             activation=F.rrelu, dropout=self.dropout, rel_emb=self.rel_emb)
            self.layers.append(h2h)


    def forward(self, g, init_ent_emb):
        node_id = g.ndata['id'].squeeze()
        g.ndata['h'] = init_ent_emb[node_id]
        for i, layer in enumerate(self.layers):
            layer(g, [])
        return g.ndata.pop('h')


class UnionRGCNLayer(nn.Module):
    def __init__(self, in_feat, out_feat, num_rels, bias=None,
                 activation=None, dropout=0.0,  rel_emb=None):
        super(UnionRGCNLayer, self).__init__()

        self.in_feat = in_feat
        self.out_feat = out_feat
        self.bias = bias
        self.activation = activation

        self.num_rels = num_rels
        self.emb_rel = rel_emb
        self.ob = None
        self.sub = None

        # WL
        self.weight_neighbor = nn.Parameter(torch.Tensor(self.in_feat, self.out_feat))
        nn.init.xavier_uniform_(self.weight_neighbor, gain=nn.init.calculate_gain('relu'))

        self.loop_weight = nn.Parameter(torch.Tensor(in_feat, out_feat))
        nn.init.xavier_uniform_(self.loop_weight, gain=nn.init.calculate_gain('relu'))
        self.evolve_loop_weight = nn.Parameter(torch.Tensor(in_feat, out_feat))
        nn.init.xavier_uniform_(self.evolve_loop_weight, gain=nn.init.calculate_gain('relu'))

        if dropout:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None

    def propagate(self, g):
        g.update_all(lambda x: self.msg_func(x), fn.sum(msg='msg', out='h'), self.apply_func)

    def forward(self, g, prev_h):
        masked_index = torch.masked_select(
            torch.arange(0, g.number_of_nodes(), dtype=torch.long).cuda(),
            (g.in_degrees(range(g.number_of_nodes())) > 0))
        loop_message = torch.mm(g.ndata['h'], self.evolve_loop_weight)
        loop_message[masked_index, :] = torch.mm(g.ndata['h'], self.loop_weight)[masked_index, :]
       
        # calculate the neighbor message with weight_neighbor
        self.propagate(g)
        node_repr = g.ndata['h']


        node_repr = node_repr + loop_message

        if self.activation:
            node_repr = self.activation(node_repr)
        if self.dropout is not None:
            node_repr = self.dropout(node_repr)
        g.ndata['h'] = node_repr
        return node_repr

    def msg_func(self, edges):
        relation = self.emb_rel.index_select(0, edges.data['type']).view(-1, self.out_feat)
        edge_type = edges.data['type']
        edge_num = edge_type.shape[0]
        node = edges.src['h'].view(-1, self.out_feat)


        msg = node + relation

        msg = torch.mm(msg, self.weight_neighbor)
        return {'msg': msg}

    def apply_func(self, nodes):
        return {'h': nodes.data['h'] * nodes.data['norm']}


class ConvTransE(torch.nn.Module):
    def __init__(self, num_entities, embedding_dim, dropout=0, channels=50, kernel_size=3):

        super(ConvTransE, self).__init__()

        self.dropout1 = torch.nn.Dropout(dropout)
        self.dropout2 = torch.nn.Dropout(dropout)
        self.dropout3 = torch.nn.Dropout(dropout)
        self.embedding_dim = embedding_dim

        self.conv = torch.nn.Conv1d(2, channels, kernel_size, 
                                    stride=1, padding=int(math.floor(kernel_size/2)))
        self.bn0 = torch.nn.BatchNorm1d(2)
        self.bn1 = torch.nn.BatchNorm1d(channels)
        self.bn2 = torch.nn.BatchNorm1d(embedding_dim)

        self.fc = torch.nn.Linear(embedding_dim * channels, embedding_dim)

    def forward(self, embedding, emb_rel, triplets):
        batch_size = len(triplets)
        e1_embedded_all = torch.tanh(embedding)
        e1_embedded = e1_embedded_all[triplets[:, 0]].unsqueeze(1)
        rel_embedded = emb_rel[triplets[:, 1]].unsqueeze(1)
        stacked_inputs = torch.cat([e1_embedded, rel_embedded], 1)
        stacked_inputs = self.bn0(stacked_inputs)
        x = self.dropout1(stacked_inputs)
        x = self.conv(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = x.view(batch_size, -1)
        x = self.fc(x)
        x = self.dropout3(x)
        if batch_size > 1:
            x = self.bn2(x)
        x = F.relu(x)
        x = torch.mm(x, e1_embedded_all.transpose(1, 0))

        return x
