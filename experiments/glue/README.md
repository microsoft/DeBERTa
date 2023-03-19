# GLUE fine-tuning task
To run the experiment, you need to

run `./mnli.sh` for fine-tuning mnli base model, 

run `./mnli.sh` for fine-tuning mnli large model.

run `./cola.sh` for fine-tuning cola large model.

run `./sst2.sh` for fine-tuning sst2 large model.

run `./stsb.sh` for fine-tuning stsb large model.

run `./rte.sh` for fine-tuning rte large model.

run `./qqp.sh` for fine-tuning qqp large model.

run `./qnli.sh` for fine-tuning qnli large model.

run `./mrpc.sh` for fine-tuning mrpc large model.

## Export model to ONNX format and quantization

To export model to onnx format during evaluation, use argument `--export_ort_model True`. 
To export quantized model, use `--fp16  False --export_ort_model True`.
The exported model will be under output folder, and end with 
`<prefix>__onnx_fp16.bin` if fp16 is True, otherwise the outputs will be `<prefix>__onnx_fp32.bin` and `<prefix>__onnx_qt.bin`.


Please check [ONNX document](https://onnxruntime.ai/docs/performance/quantization.html) for more details.
