import json
import copy

__all__=['AbsModelConfig', 'ModelConfig']

class AbsModelConfig(object):
    def __init__(self):
        pass

    @classmethod
    def from_dict(cls, json_object):
        """Constructs a `ModelConfig` from a Python dictionary of parameters."""
        config = cls()
        for key, value in json_object.items():
            if isinstance(value, dict):
                value = AbsModelConfig.from_dict(value)
            config.__dict__[key] = value
        return config

    @classmethod
    def from_json_file(cls, json_file):
        """Constructs a `ModelConfig` from a json file of parameters."""
        with open(json_file, "r", encoding='utf-8') as reader:
            text = reader.read()
        return cls.from_dict(json.loads(text))

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        def _json_default(obj):
            if isinstance(obj, AbsModelConfig):
                return obj.__dict__
        return json.dumps(self.__dict__, indent=2, sort_keys=True, default=_json_default) + "\n"

class ModelConfig(AbsModelConfig):
    """Configuration class to store the configuration of a `BertModel`.
    """
    def __init__(self):
        """Constructs ModelConfig.

        Args:
            vocab_size_or_config_json_file: Vocabulary size of `inputs_ids` in `BertModel`.
            hidden_size: Size of the encoder layers and the pooler layer.
            num_hidden_layers: Number of hidden layers in the Transformer encoder.
            num_attention_heads: Number of attention heads for each attention layer in
                the Transformer encoder.
            intermediate_size: The size of the "intermediate" (i.e., feed-forward)
                layer in the Transformer encoder.
            hidden_act: The non-linear activation function (function or string) in the
                encoder and pooler. If string, "gelu", "relu" and "swish" are supported.
            hidden_dropout_prob: The dropout probabilitiy for all fully connected
                layers in the embeddings, encoder, and pooler.
            attention_probs_dropout_prob: The dropout ratio for the attention
                probabilities.
            max_position_embeddings: The maximum sequence length that this model might
                ever be used with. Typically set this to something large just in case
                (e.g., 512 or 1024 or 2048).
            type_vocab_size: The vocabulary size of the `token_type_ids` passed into
                `BertModel`.
            initializer_range: The sttdev of the truncated_normal_initializer for
                initializing all weight matrices.
        """
        self.hidden_size = 768
        self.num_hidden_layers = 12
        self.num_attention_heads = 12
        self.hidden_act = "gelu"
        self.intermediate_size = 3072
        self.hidden_dropout_prob = 0.1
        self.attention_probs_dropout_prob = 0.1
        self.max_position_embeddings = 512
        self.type_vocab_size = 0
        self.initializer_range = 0.02
        self.layer_norm_eps = 1e-7
        self.padding_idx = 0
        self.vocab_size = -1
