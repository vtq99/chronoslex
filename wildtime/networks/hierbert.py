from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import numpy as np
from torch import nn
from transformers import AutoModel
from transformers.file_utils import ModelOutput


@dataclass
class SimpleOutput(ModelOutput):
    last_hidden_state: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None


def sinusoidal_init(num_embeddings: int, embedding_dim: int):
    # keep dim 0 for padding token position encoding zero vector
    position_enc = np.array([
        [pos / np.power(10000, 2 * i / embedding_dim) for i in range(embedding_dim)]
        if pos != 0 else np.zeros(embedding_dim) for pos in range(num_embeddings)])

    position_enc[1:, 0::2] = np.sin(position_enc[1:, 0::2])  # dim 2i
    position_enc[1:, 1::2] = np.cos(position_enc[1:, 1::2])  # dim 2i+1
    return torch.from_numpy(position_enc).type(torch.FloatTensor)


class HierarchicalBert(nn.Module):

    def __init__(self, num_classes):
        super(HierarchicalBert, self).__init__()
        self.num_classes = num_classes
        # Pre-trained segment (token-wise) encoder, e.g., BERT
        self.bert = AutoModel.from_pretrained("nlpaueb/legal-bert-base-uncased")
        self.config = self.bert.config
        # Specs for the segment-wise encoder
        self.hidden_size = self.bert.config.hidden_size
        self.max_segments = 64
        self.max_segment_length = 128
        # Init sinusoidal positional embeddings
        self.seg_pos_embeddings = nn.Embedding(self.max_segments + 1, self.bert.config.hidden_size,
                                               padding_idx=0,
                                               _weight=sinusoidal_init(self.max_segments + 1,
                                                                       self.bert.config.hidden_size))
        # Init segment-wise transformer-based encoder
        self.seg_encoder = nn.Transformer(d_model=self.bert.config.hidden_size,
                                          nhead=self.bert.config.num_attention_heads,
                                          batch_first=True, dim_feedforward=self.bert.config.intermediate_size,
                                          activation=self.bert.config.hidden_act,
                                          dropout=self.bert.config.hidden_dropout_prob,
                                          layer_norm_eps=self.bert.config.layer_norm_eps,
                                          num_encoder_layers=2, num_decoder_layers=0).encoder

        self.classifier = nn.Linear(self.config.hidden_size, self.num_classes)

    def forward(self, x):
        # Hypothetical Example
        # Batch of 4 documents: (batch_size, n_segments, max_segment_length) --> (4, 64, 128)
        # BERT-BASE encoder: 768 hidden units
        input_ids = x[:, :, :, 0]
        attention_mask = x[:, :, :, 1]
        token_type_ids = x[:, :, :, 2]

        # Squash samples and segments into a single axis (batch_size * n_segments, max_segment_length) --> (256, 128)
        input_ids_reshape = input_ids.contiguous().view(-1, input_ids.size(-1))
        attention_mask_reshape = attention_mask.contiguous().view(-1, attention_mask.size(-1))
        if token_type_ids is not None:
            token_type_ids_reshape = token_type_ids.contiguous().view(-1, token_type_ids.size(-1))
        else:
            token_type_ids_reshape = None

        # Encode segments with BERT --> (256, 128, 768)
        encoder_outputs = self.bert(input_ids=input_ids_reshape,
                                    attention_mask=attention_mask_reshape,
                                    token_type_ids=token_type_ids_reshape)[0]

        # Reshape back to (batch_size, n_segments, max_segment_length, output_size) --> (4, 64, 128, 768)
        encoder_outputs = encoder_outputs.contiguous().view(input_ids.size(0), self.max_segments,
                                                            self.max_segment_length,
                                                            self.hidden_size)

        # Gather CLS outputs per segment --> (4, 64, 768)
        encoder_outputs = encoder_outputs[:, :, 0]

        # Infer real segments, i.e., mask paddings
        seg_mask = (torch.sum(input_ids, 2) != 0).to(input_ids.dtype)
        # Infer and collect segment positional embeddings
        seg_positions = torch.arange(1, self.max_segments + 1).to(input_ids.device) * seg_mask
        # Add segment positional embeddings to segment inputs
        encoder_outputs += self.seg_pos_embeddings(seg_positions)

        # Encode segments with segment-wise transformer
        seg_encoder_outputs = self.seg_encoder(encoder_outputs)

        # Collect document representation
        outputs, _ = torch.max(seg_encoder_outputs, 1)

        # Compute label scores / outputs
        return self.classifier(outputs)

        # return SimpleOutput(last_hidden_state=outputs, hidden_states=outputs)
