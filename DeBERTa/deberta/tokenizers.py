#
# Author: penhe@microsoft.com
# Date: 04/25/2019
#

""" tokenizers
"""

from .gpt2_tokenizer import GPT2Tokenizer
from .spm_tokenizer import *

__all__ = ['tokenizers']
tokenizers={
    'gpt2': GPT2Tokenizer,
    'spm': SPMTokenizer
    }
