import numpy as np
import torch
from collections import defaultdict
import scipy.sparse as sp
import dgl
import dgl.function as fn

# For Loading Data
def split_by_time(data):
    snapshot_list = []
    snapshot = []
    snapshots_num = 0
    latest_t = 0
    for i in range(len(data)):
        t = data[i][3]
        train = data[i]
        if latest_t != t: 
            latest_t = t
            if len(snapshot):
                snapshot_list.append(np.array(snapshot).copy())
                snapshots_num += 1
            snapshot = []
        snapshot.append(train[:3])
    if len(snapshot) > 0:
        snapshot_list.append(np.array(snapshot).copy())
        snapshots_num += 1

    union_num = [1]
    nodes = []
    rels = []
    for snapshot in snapshot_list:
        uniq_v, edges = np.unique((snapshot[:,0], snapshot[:,2]), return_inverse=True)  # relabel
        uniq_r = np.unique(snapshot[:,1])
        edges = np.reshape(edges, (2, -1))
        nodes.append(len(uniq_v))
        rels.append(len(uniq_r)*2)
    return snapshot_list

def load_all_answers_for_time_filter(total_data, num_rels, num_nodes):
    all_ans_list = []
    all_snap = split_by_time(total_data)
    for snap in all_snap:
        all_ans_t = load_all_answers_for_filter(snap, num_rels)
        all_ans_list.append(all_ans_t)
    return all_ans_list

def load_all_answers_for_filter(total_data, num_rel):
    all_ans = {}
    for line in total_data:
        s, r, o = line[: 3]
        add_subject(s, o, r, all_ans, num_rel=num_rel)
        add_object(s, o, r, all_ans, num_rel=0)
    return all_ans

def add_subject(e1, e2, r, d, num_rel):
    if not e2 in d:
        d[e2] = {}
    if not r+num_rel in d[e2]:
        d[e2][r+num_rel] = set()
    d[e2][r+num_rel].add(e1)

def add_relation(e1, e2, r, d):
    if not e1 in d:
        d[e1] = {}
    if not e2 in d[e1]:
        d[e1][e2] = set()
    d[e1][e2].add(r)

def add_object(e1, e2, r, d, num_rel):
    if not e1 in d:
        d[e1] = {}
    if not r in d[e1]:
        d[e1][r] = set()
    d[e1][r].add(e2)


# For entity related relation
def get_entity_related_relation(snap, num_nodes, num_rels):
    weight, row, col = [],[],[]
    d = defaultdict(list)
    for triple in snap:
        d[triple[0]].append(triple[1])
    for i in range(num_nodes):
        for j in d[i]:
            weight.append(1/len(d[i]))
            row.append(i)
            col.append(j)
    return sp.csr_matrix((weight, (row, col)), shape=(num_nodes, num_rels*2))

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape).cuda()

# for data inverse
def add_inverse(snap_list, num_rel):
    # add inverse triples to train_list
    all_list = []
    for triples in snap_list:
        inverse_triples = triples[:, [2, 1, 0]]
        inverse_triples[:, 1] = inverse_triples[:, 1] + num_rel
        all_triples = np.concatenate((triples, inverse_triples))
        all_list.append(all_triples)
    return all_list

def get_inverse(snap_list, num_rel):
    all_list = []
    for triples in snap_list:
        inverse_triples = triples[:, [2, 1, 0]]
        inverse_triples[:, 1] = inverse_triples[:, 1] + num_rel
        all_list.append(inverse_triples)
    return all_list



# for dgl graph
def build_sub_graph(num_nodes, num_rels, triples, gpu):

    def comp_deg_norm(g):
        in_deg = g.in_degrees(range(g.number_of_nodes())).float()
        in_deg[torch.nonzero(in_deg == 0).view(-1)] = 1
        norm = 1.0 / in_deg
        return norm

    src, rel, dst = triples.transpose()

    g = dgl.graph((src,dst),num_nodes = num_nodes)
    norm = comp_deg_norm(g)
    node_id = torch.arange(0, num_nodes, dtype=torch.long).view(-1, 1)
    g.ndata.update({'id': node_id, 'norm': norm.view(-1, 1)})
    g.apply_edges(lambda edges: {'norm': edges.dst['norm'] * edges.src['norm']})
    g.edata['type'] = torch.LongTensor(rel)

    g.to(gpu)
    return g