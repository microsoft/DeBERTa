# DeBERTa: Fine-tuning with ONNX Runtime.

## Requirements
- All the DeBERTA requirements
- onnx
- onnxruntime

### Workaround fixes
- The workaround is needed until MSE operator becomes available in ORT
  vi $PYTHONPATH/site-packages/torch/nn/functional.py
  search for "def mse_loss"
  proceed to lines
     else:
        expanded_input, expanded_target = torch.broadcast_tensors(input, target)
        ret = torch._C._nn.mse_loss(expanded_input, expanded_target, _Reduction.get_enum(reduction))
  and change them to:
        expanded_input = input
        expanded_target = target
        t = expanded_input - expanded_target
        t = t ** 2
        ret = torch.mean(t)

- The workaround is needed until fix is available to disable Unsqueeze optimization for trainable weights in ORT
  Changes in onnx runtime code:
  Open onnxruntime/onnxruntime/core/graph/graph_utils.cc 
  Hardcode to return false to disable Unsqueeze optimization for DeBERTa, see below 
  if (output_name_is_changing) {
    std::vector<GraphEdge> output_edges = GetNodeOutputEdges(node);
    can_remove = CanUpdateImplicitInputNameInSubgraphs(graph, output_edges, initializer_name, logger);
    can_remove = false;  // <- Put this line in

## Run task

``` bash
task=STS-B 
OUTPUT=/tmp/DeBERTa/exps/$task
python3 -m DeBERTa.apps.orttrain --task_name $task \
  --data_dir $cache_dir/glue_tasks/$task \
  --output_dir $OUTPUT \
  --eval_batch_size 128 \
  --train_batch_size 32 \
  --num_train_epochs 6 \
  --learning_rate 2e-5 \
  --max_seq_len 128 \
  --init_model base \
  --seed 123
```