# adapted from run_glue.py of huggingface transformers

import dataclasses
import logging
import os
from dataclasses import dataclass, field
from typing import Dict, Optional
import unittest
import numpy as np
from numpy.testing import assert_allclose

from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EvalPrediction,
    GlueDataset,
    GlueDataTrainingArguments,
    TrainingArguments,
    glue_compute_metrics,
    glue_output_modes,
    glue_tasks_num_labels,
    set_seed,
)

import onnxruntime
from onnxruntime.capi.ort_trainer import ORTTrainer, LossScaler, ModelDescription, IODescription
from onnxruntime.capi._pybind_state import get_mpi_context_local_rank, get_mpi_context_local_size, get_mpi_context_world_rank, get_mpi_context_world_size

from .orttraining_transformer_trainer import ORTTransformerTrainer

import torch

logger = logging.getLogger(__name__)

class ORTGlueTest(unittest.TestCase):

    def setUp(self, args):
        # configurations not to be changed accoss tests
        self.max_seq_length = args.max_seq_length
        self.train_batch_size = args.train_batch_size
        self.eval_batch_size = args.eval_batch_size
        self.learning_rate = args.learning_rate
        self.num_train_epochs = args.num_train_epochs
        self.local_rank = -1
        self.world_size = 1
        self.overwrite_output_dir = True
        self.gradient_accumulation_steps = 1
        self.data_dir = args.data_dir
        self.output_dir = args.output_dir
        self.cache_dir = args.cache_dir
        self.logging_steps = 100
        self.rtol = 1e-02
        self.seed = args.seed

    def model_to_desc(self):
        batch_size = int(self.train_batch_size) # * self.world_size)
        new_model_desc = {
            'inputs': [
                ('input_ids', ['batch', 'max_seq_len_in_batch'],),
                ('token_type_ids', ['batch', 'max_seq_len_in_batch'],),
                ('attention_mask', ['batch', 'max_seq_len_in_batch'],),
                ('labels', ['batch', ],)],
            'outputs': [('loss', [], True),
                        ('logits', ['batch',])]}
        model_desc = ModelDescription([
            IODescription('input_ids', ['batch', 'max_seq_len_in_batch']),
            IODescription('token_type_ids', ['batch', 'max_seq_len_in_batch']),
            #IODescription('position_ids', [batch_size, self.max_seq_length]),
            IODescription('attention_mask', ['batch', 'max_seq_len_in_batch']),
            IODescription('labels', ['batch',])], [
            IODescription('loss', []),
            IODescription('logits', ['batch',])])

        return model_desc, new_model_desc

    def run_glue(self, task_name, fp16, use_new_api):
        data_args = GlueDataTrainingArguments(
            task_name=task_name, data_dir=os.path.join(self.data_dir, task_name),
            max_seq_length=self.max_seq_length)

        training_args = TrainingArguments(
            output_dir=os.path.join(self.output_dir, task_name), do_train=True, do_eval=True,
            per_gpu_train_batch_size=self.train_batch_size,
            per_gpu_eval_batch_size = self.eval_batch_size,
            learning_rate=self.learning_rate, num_train_epochs=self.num_train_epochs,
            local_rank=self.local_rank,
            overwrite_output_dir=self.overwrite_output_dir, gradient_accumulation_steps=self.gradient_accumulation_steps,
            fp16=fp16, logging_steps=self.logging_steps,
            seed=self.seed)

        # Setup logging
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
        )
        logger.warning(
            "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
            training_args.local_rank,
            training_args.device,
            training_args.n_gpu,
            bool(training_args.local_rank != -1),
            training_args.fp16,
        )
        logger.info("Training/evaluation parameters %s", training_args)

        set_seed(training_args.seed)
        onnxruntime.set_seed(training_args.seed)

        try:
            num_labels = glue_tasks_num_labels[data_args.task_name]
            output_mode = glue_output_modes[data_args.task_name]
        except KeyError:
            raise ValueError("Task not found: %s" % (data_args.task_name))

        train_dataset = (
            GlueDataset(data_args, tokenizer=self.tokenizer)
            if training_args.do_train
            else None
        )

        eval_dataset = (
            GlueDataset(data_args, tokenizer=self.tokenizer, mode="dev")
            if training_args.do_eval
            else None
        )

        def compute_metrics(p: EvalPrediction) -> Dict:
            if output_mode == "classification":
                preds = np.argmax(p.predictions, axis=1)
            elif output_mode == "regression":
                preds = np.squeeze(p.predictions)
            return glue_compute_metrics(data_args.task_name, preds, p.label_ids)

        model_desc, new_model_desc = self.model_to_desc()
        # Initialize the ORTTrainer within ORTTransformerTrainer
        trainer = ORTTransformerTrainer(
            model=self.model,
            model_desc=model_desc,
            new_model_desc=new_model_desc,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=compute_metrics,
            use_new_api=use_new_api,
            world_size=self.world_size,
        )

        # Training
        if training_args.do_train:
            trainer.train()
            trainer.save_model()

        # Evaluation
        results = {}
        if training_args.do_eval and training_args.local_rank in [-1, 0]:
            logger.info("*** Evaluate ***")

            result = trainer.evaluate()

            logger.info("***** Eval results {} *****".format(data_args.task_name))
            for key, value in result.items():
               logger.info("  %s = %s", key, value)

            results.update(result)

        return results