# Best Trial Latency Breakdown — entwined_ring/s7/b4/supermesh_alter:flat_butterfly

## Configuration
- Partition: `entwined_ring` / shards=7
- Batch size: 4
- NoC: `supermesh_alter:xy`
- NoW: `flat_butterfly:xy`
- Tile pipeline: False

## Total Latency (max-path / stage-overlap model)
- **Total**: 209,407 cycles
- FFN path: 101,207 cycles
- Comm tail: 108,200 cycles

## FFN Path Detail
```
W1 = max(compute, mem) = max(31,382, 49,843) = 49,843
W3 = max(compute, mem) = max(31,382, 49,843) = 49,843
ElemMul = 1,756
W2 = max(compute, mem) = max(30,742, 49,608) = 49,608
ffn_path = max(W1,W3) + Elem + W2 = max(49,843,49,843) + 1,756 + 49,608 = 101,207
```

## Communication Tail
```
comm_tail = max(network, io, allreduce)
         = max(1,076, 1,355, 108,200)
         = 108,200
```

## Verification
total = ffn_path + comm_tail = 101,207 + 108,200 = 209,407