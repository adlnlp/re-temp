# Re-Temp: Relation-Aware Temporal Representation Learning for Temporal Knowledge Graph Completion
<h3 align="center"><strong>Kunze Wang, Caren Han, Josiah Poon</strong></h3>

This is the official implementation for [Re-Temp: Relation-Aware Temporal Representation Learning for Temporal Knowledge Graph Completion](https://aclanthology.org/2023.findings-emnlp.20/) (EMNLP Findings 2023).

## One Line Running
Simply run `python main.py -d ICEWS14`

*Requirments*：torch, dgl(cuda version)

## Arguments explanation
| Argument     | Default   | Description |
| ----------- | ----------- |----------- |
| --gpu | 0 | Set cuda device |
| --dataset  -d | ICEWS14 | Dataset used here, please choose from GDELT, ICEWS05-15, ICEWS14, ICEWS14s, ICEWS18, WIKI|
| --dropout | 0.2 | Dropout rate |
| --n-hidden  | 200 | Layer hidden dimension |
| --n-layers  | 2 | Number of layers of one GNN |
| --history-len  | 3 | history length |
| --lr | 0.001 |learning rate |
| --early_stop  | 5 | early stop epochs |
| --easy_copy  | 0 | Remove most print results, only keep the final output |


## Citation
```
@inproceedings{wang2023re,
  title={Re-Temp: Relation-Aware Temporal Representation Learning for Temporal Knowledge Graph Completion},
  author={Wang, Kunze and Han, Caren and Poon, Josiah},
  booktitle={Findings of the Association for Computational Linguistics: EMNLP 2023},
  pages={258--269},
  year={2023}
}
```

## Acknowledgement
Some of the codes are inspired by:
- https://github.com/Lee-zix/CEN
- https://github.com/INK-USC/RE-Net
We express gratitude for all the previous contributions in this area.
