from .glue_tasks import *
from .ner_task import *
from .race_task import *

tasks={
    'mnli': MNLITask,
    'anli': ANLITask,
    'sts-b': STSBTask,
    'sst-2': SST2Task,
    'qqp': QQPTask,
    'cola': ColaTask,
    'mrpc': MRPCTask,
    'rte': RTETask,
    'qnli': QNLITask,
    'race': RACETask,
    'ner': NERTask
    }
