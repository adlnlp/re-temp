import torch
from utils import *

def test_model(model, history_len, history_list, test_list1, test_list2, num_rels, 
         num_nodes, all_ans_list, model_name, args, err_mat_list, err_inv_mat_list,mode):

    ranks_raw, ranks_filter, mrr_raw_list, mrr_filter_list = [], [], [], []

    idx = 0
    start_time = len(history_list)
    if mode == "test":
        # test mode: load parameter form file
        checkpoint = torch.load(model_name, map_location=torch.device(args.gpu))
        if not args.easy_copy:
            print("Load Model name: {}. Using best epoch : {}".format(model_name, checkpoint['epoch']))  # use best stat checkpoint
            print("-"*10+"start testing"+"-"*10)
        model.load_state_dict(checkpoint['state_dict'])

    model.eval()

    input_list = [snap for snap in history_list[-history_len:]]

    for time_idx in range(len(test_list1)):
        tc = start_time + time_idx
        history_glist = [build_sub_graph(num_nodes, num_rels, g, args.gpu) for g in input_list]
        
        tlist = list(range(tc - history_len,tc))
        # tlist = [min(start_time-args.start_history_len-1,t) for t in tlist]
        tlist = torch.Tensor(tlist).cuda() 

        test_snap1 = test_list1[time_idx]
        test_snap2 = test_list2[time_idx]
        test_snap = np.concatenate([test_snap1,test_snap2])

        test_triples_input1 = torch.LongTensor(test_snap1).cuda()
        test_triples_input2 = torch.LongTensor(test_snap2).cuda()
        test_triples_input = torch.cat([test_triples_input1,test_triples_input2])

        err_mat1 = sparse_mx_to_torch_sparse_tensor(err_mat_list[tc])
        err_mat2 = sparse_mx_to_torch_sparse_tensor(err_inv_mat_list[tc])

        final_score = model.predict(history_glist,
                                    tlist, 
                                    test_triples_input1, 
                                    test_triples_input2, 
                                    err_mat1, 
                                    err_mat2)
        

        mrr_filter_snap, mrr_snap, rank_raw, rank_filter = get_total_rank(test_triples_input, 
                                                                          final_score, 
                                                                          all_ans_list[tc], 
                                                                          eval_bz=1000)

        # used to global statistic
        ranks_raw.append(rank_raw)
        ranks_filter.append(rank_filter)
        # used to show slide results
        mrr_raw_list.append(mrr_snap)
        mrr_filter_list.append(mrr_filter_snap)

        # reconstruct history graph list
        input_list.pop(0)
        input_list.append(test_snap)
        idx += 1
    
    mrr_raw, hit_res_raw = stat_ranks(ranks_raw, "raw_ent")
    mrr_filter, hit_res_filter = stat_ranks(ranks_filter, "filter_ent")

    return mrr_raw, mrr_filter, hit_res_filter


def get_total_rank(test_triples, score, all_ans, eval_bz):
    num_triples = len(test_triples)
    n_batch = (num_triples + eval_bz - 1) // eval_bz
    rank = []
    filter_rank = []
    for idx in range(n_batch):
        batch_start = idx * eval_bz
        batch_end = min(num_triples, (idx + 1) * eval_bz)
        triples_batch = test_triples[batch_start:batch_end, :]
        score_batch = score[batch_start:batch_end, :]

        target = test_triples[batch_start:batch_end, 2]
        rank.append(sort_and_rank(score_batch, target))

        filter_score_batch = filter_score(triples_batch, score_batch, all_ans)
        filter_rank.append(sort_and_rank(filter_score_batch, target))

    rank = torch.cat(rank)
    filter_rank = torch.cat(filter_rank)
    rank += 1 # change to 1-indexed
    filter_rank += 1
    mrr = torch.mean(1.0 / rank.float())
    filter_mrr = torch.mean(1.0 / filter_rank.float())
    return filter_mrr.item(), mrr.item(), rank, filter_rank

def stat_ranks(rank_list, method):
    hits = [1, 3, 10]
    hit_res = []
    total_rank = torch.cat(rank_list)

    mrr = torch.mean(1.0 / total_rank.float())

    for hit in hits:
        avg_count = torch.mean((total_rank <= hit).float())
        hit_res.append(avg_count)
        
    return mrr.item(), hit_res

def sort_and_rank(score, target):
    _, indices = torch.sort(score, dim=1, descending=True)
    indices = torch.nonzero(indices == target.view(-1, 1))
    indices = indices[:, 1].view(-1)
    return indices

def filter_score(test_triples, score, all_ans):
    if all_ans is None:
        return score
    test_triples = test_triples.cpu()
    for _, triple in enumerate(test_triples):
        h, r, t = triple
        ans = list(all_ans[h.item()][r.item()])
        ans.append(h.item())
        ans.remove(t.item())
        ans = torch.LongTensor(ans)
        score[_][ans] = -10000000  
    return score
