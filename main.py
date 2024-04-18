import numpy as np
import time
import os
import random
import argparse
from tqdm import tqdm

from dataset import TKGCDataset
from utils import *
from model import DQGCN
from test import test_model


parser = argparse.ArgumentParser()

parser.add_argument("--gpu", type=int, default=0,
                    help="gpu")
parser.add_argument("-d", "--dataset", type=str, default='ICEWS14',
                    help="dataset to use")


# model hyperparamters
parser.add_argument("--dropout", type=float, default=0.2,
                    help="dropout probability")
parser.add_argument("--n-hidden", type=int, default=200,
                    help="number of hidden units")
parser.add_argument("--n-layers", type=int, default=2,
                    help="number of propagation rounds")
parser.add_argument("--history-len", type=int, default=3,
                help="start history length")
parser.add_argument("--alpha", type=float, default=0.5,
                    help="xxx")
# parser.add_argument("--beta", type=float, default=0.1,
                    # help="xxx")


# training hyperparameters
parser.add_argument("--n-epochs", type=int, default=30,
                    help="number of minimum training epochs on each time step")
parser.add_argument("--lr", type=float, default=0.001,
                    help="learning rate")
parser.add_argument("--early_stop", type=int, default=5,
                    help="number of minimum training epochs on each time step")

parser.add_argument("--easy_copy", type=float, default=0,
                    help="if 0, print all outputs")

args = parser.parse_args()
print(args)

# Load Data
data = TKGCDataset("data/"+args.dataset)
data.load()
train_list = split_by_time(data.train)
valid_list = split_by_time(data.valid)
test_list = split_by_time(data.test)
total_data = np.concatenate((data.train, data.valid, data.test), axis=0)

if not args.easy_copy:
    print(data.dir)
    print("# Sanity Check:  entities: {}".format(data.num_nodes))
    print("# Sanity Check:  relations: {}".format(data.num_rels))
    print("# Sanity Check:  edges: {}".format(len(data.train)))
    print("total data length ", len(total_data))


# Load Answers
num_nodes = data.num_nodes
num_rels = data.num_rels
all_ans_list = load_all_answers_for_time_filter(total_data, num_rels, num_nodes)

# Get entity related relation matrix 
err_mat_list = []
for snap in train_list+valid_list+test_list:
    err_mat_list.append(get_entity_related_relation(snap, num_nodes, num_rels))

# add inverse and get inverse
train_inv_list = get_inverse(train_list, num_rels)
train_list = add_inverse(train_list, num_rels) # train list modified
valid_inv_list = get_inverse(valid_list, num_rels)
test_inv_list = get_inverse(test_list, num_rels)

# Get inversed entity related relation matrix 
err_inv_mat_list = []
for snap in train_inv_list+valid_inv_list+test_inv_list:
    err_inv_mat_list.append(get_entity_related_relation(snap, num_nodes, num_rels))


# Get model
model = DQGCN(num_nodes, num_rels, args)
torch.cuda.set_device(args.gpu)
model.cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)

model_name = str(int(time.time()))
if not os.path.exists('../models/{}/'.format(args.dataset)):
    os.makedirs('../models/{}/'.format(args.dataset))
model_state_file = '../models/{}/{}'.format(args.dataset, model_name) 
if not args.easy_copy:
    print("Model name : {}".format(model_state_file))


# Train the model
import time
best_mrr = 0
best_epoch = 0
for epoch in tqdm(range(args.n_epochs)):
    tt = time.time()
    model.train()
    losses = []

    idx = [_ for _ in range(len(train_list))]
    random.shuffle(idx)
    for i in idx:
        if i == 0 or i == 1: continue
        if i - args.history_len<0:
            input_list = train_list[0: i]
            tlist = torch.Tensor(list(range(len(input_list)))).cuda()
        else:
            input_list = train_list[i-args.history_len: i]
            tlist = torch.Tensor(list(range(i-args.history_len,i))).cuda()
        
        history_glist = [build_sub_graph(num_nodes, num_rels, snap, args.gpu) for snap in input_list]

        output = train_list[i]
        output1 = torch.from_numpy(output[:len(output)//2]).long().cuda()
        output2 = torch.from_numpy(output[len(output)//2:]).long().cuda()

        err_mat = sparse_mx_to_torch_sparse_tensor(err_mat_list[i])
        err_inv_mat = sparse_mx_to_torch_sparse_tensor(err_inv_mat_list[i])

        loss1= model.get_loss(history_glist, tlist, output1, err_mat)
        loss2= model.get_loss(history_glist, tlist, output2, err_inv_mat)
        loss = loss1+loss2
        losses.append(loss.item())

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  
        optimizer.step()
        optimizer.zero_grad()
    if not args.easy_copy:    
        print("His {:04d}, Epoch {:04d} | Ave Loss: {:.4f} | Best MRR {:.4f} "
            .format(args.history_len, epoch, np.mean(losses), best_mrr))
    
    # validation        
    mrr_raw, mrr_filter, hit_res_filter= test_model(model,
                                        args.history_len, 
                                        train_list, 
                                        valid_list, 
                                        valid_inv_list,
                                        num_rels, 
                                        num_nodes,  
                                        all_ans_list, 
                                        model_state_file, 
                                        args,
                                        err_mat_list,
                                        err_inv_mat_list,
                                        mode="train")
    if not args.easy_copy:
        print("MRR : {:.6f}".format(mrr_filter))
        print("Hits @ 1: {:.6f}".format(hit_res_filter[0]))
        print("Hits @ 3: {:.6f}".format(hit_res_filter[1]))
        print("Hits @ 10: {:.6f}".format(hit_res_filter[2]))


    if mrr_filter< best_mrr:
        if epoch >= args.n_epochs or epoch - best_epoch > args.early_stop:
            if not args.easy_copy:
                print('early stop!')
            break
    else:
        best_mrr = mrr_filter
        best_epoch = epoch
        torch.save({'state_dict': model.state_dict(), 'epoch': epoch}, model_state_file)
    if not args.easy_copy:
        print("Time taken: {:.2f}s".format(time.time()-tt))


# Testing
valid_list = add_inverse(valid_list, num_rels) # Before testing, add inverse for validation set

mrr_raw, mrr_filter, hit_res_filter= test_model(model, 
                                    args.history_len, 
                                    train_list+valid_list,
                                    test_list, 
                                    test_inv_list,
                                    num_rels, 
                                    num_nodes,  
                                    all_ans_list, 
                                    model_state_file, 
                                    args,
                                    err_mat_list,
                                    err_inv_mat_list,
                                    mode="test")
if not args.easy_copy:
    print("MRR : {:.6f}".format(mrr_filter))
    print("Hits @ 1: {:.6f}".format(hit_res_filter[0]))
    print("Hits @ 3: {:.6f}".format(hit_res_filter[1]))
    print("Hits @ 10: {:.6f}".format(hit_res_filter[2]))
else:
    print("{:.2f}".format(mrr_filter*100))
    print("{:.2f}".format(hit_res_filter[0]*100))
    print("{:.2f}".format(hit_res_filter[1]*100))
    print("{:.2f}".format(hit_res_filter[2]*100))
