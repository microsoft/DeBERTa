# SiFT (Scale Invariant Fine-Tuning) 

## Usage

For example to try SiFT in DeBERTa, please check `experiments/glue/mnli.sh base-sift` or `experiments/glue/mnli.sh xxlarge-v2-sift`


Here is an example to consume SiFT in your existing code,

  ```python
  # Create DeBERTa model
  adv_modules = hook_sift_layer(model, hidden_size=768)
  adv = AdversarialLearner(model, adv_modules)
  def logits_fn(model, *wargs, **kwargs):
    logits,_ = model(*wargs, **kwargs)
    return logits
  logits,loss = model(**data)

  loss = loss + adv.loss(logits, logits_fn, **data)
  # Other steps is the same as general training.

  ```

## Ablation study results


| Model                     |  MNLI-m/mm   | SST-2 | QNLI | CoLA | RTE    | MRPC  | QQP   |STS-B |
|---------------------------|-------------|-------|------|------|--------|-------|-------|------|
|                           |  Acc         | Acc   | Acc  | MCC  | Acc    |Acc/F1 |Acc/F1 |P/S   |
|**[DeBERTa-V2-XXLarge](https://huggingface.co/microsoft/deberta-v2-xxlarge)<sup>1,2</sup>**|91.7/91.9|97.2|96.0|72.0| 93.5| **93.1/94.9**|92.7/90.3 |93.2/93.1 |
|**[DeBERTa-V2-XXLarge](https://huggingface.co/microsoft/deberta-v2-xxlarge)<sup>1,2</sup>**|**92.0/92.1**|97.5|**96.5**|**73.5**| **96.5**| - |**93.0/90.7** | - |
