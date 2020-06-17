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
    """Configuration class to store the configuration of a :class:`~DeBERTa.deberta.DeBERTa` model.

        Attributes:
            hidden_size (int): Size of the encoder layers and the pooler layer, default: `768`.
            num_hidden_layers (int): Number of hidden layers in the Transformer encoder, default: `12`.
            num_attention_heads (int): Number of attention heads for each attention layer in
                the Transformer encoder, default: `12`.
            intermediate_size (int): The size of the "intermediate" (i.e., feed-forward)
                layer in the Transformer encoder, default: `3072`.
            hidden_act (str): The non-linear activation function (function or string) in the
                encoder and pooler. If string, "gelu", "relu" and "swish" are supported, default: `gelu`.
            hidden_dropout_prob (float): The dropout probabilitiy for all fully connected
                layers in the embeddings, encoder, and pooler, default: `0.1`.
            attention_probs_dropout_prob (float): The dropout ratio for the attention
                probabilities, default: `0.1`.
            max_position_embeddings (int): The maximum sequence length that this model might
                ever be used with. Typically set this to something large just in case
                (e.g., 512 or 1024 or 2048), default: `512`.
            type_vocab_size (int): The vocabulary size of the `token_type_ids` passed into
                `DeBERTa` model, default: `-1`.
            initializer_range (int): The sttdev of the _normal_initializer for
                initializing all weight matrices, default: `0.02`.
            relative_attention (:obj:`bool`): Whether use relative position encoding, default: `False`.
            max_relative_positions (int): The range of relative positions [`-max_position_embeddings`, `max_position_embeddings`], default: -1, use the same value as `max_position_embeddings`. 
            padding_idx (int): The value used to pad input_ids, default: `0`.
            position_biased_input (:obj:`bool`): Whether add absolute position embedding to content embedding, default: `True`.
            pos_att_type (:obj:`str`): The type of relative position attention, it can be a combination of [`p2c`, `c2p`, `p2p`], e.g. "p2c", "p2c|c2p", "p2c|c2p|p2p"., default: "None".


    """
    def __init__(self):
        """Constructs ModelConfig.

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
