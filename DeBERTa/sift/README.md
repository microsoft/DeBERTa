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

# Citation
```
@misc{he2020deberta,
    title={DeBERTa: Decoding-enhanced BERT with Disentangled Attention},
    author={Pengcheng He and Xiaodong Liu and Jianfeng Gao and Weizhu Chen},
    year={2020},
    eprint={2006.03654},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}

@article{Jiang_2020,
   title={SMART: Robust and Efficient Fine-Tuning for Pre-trained Natural Language Models through Principled Regularized Optimization},
   url={http://dx.doi.org/10.18653/v1/2020.acl-main.197},
   DOI={10.18653/v1/2020.acl-main.197},
   journal={Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics},
   publisher={Association for Computational Linguistics},
   author={Jiang, Haoming and He, Pengcheng and Chen, Weizhu and Liu, Xiaodong and Gao, Jianfeng and Zhao, Tuo},
   year={2020}
}
```
