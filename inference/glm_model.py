import torch
import torch.nn as nn
from transformers import RobertaModel
from transformers.models.roberta.modeling_roberta import *
from transformers.utils import ModelOutput
from dataclasses import dataclass
from typing import Optional, Tuple, Union

@dataclass
class MultiPredMaskedLMOutput(ModelOutput):
    """
    Output class for multi-prediction masked language modeling.

    Attributes:
        logits_all_preds (torch.FloatTensor): Logits for all predictions.
        probs (Optional[torch.FloatTensor]): Predicted class probabilities.
        last_hidden_state (Optional[Tuple[torch.FloatTensor]]): Final hidden layer output from the model.
        contacts (Optional[torch.FloatTensor]): Contact prediction matrix (e.g., for protein residues).
        all_hidden_states (Optional[Tuple[torch.FloatTensor]]): All hidden layer states, if requested.
    """
    logits_all_preds: torch.FloatTensor = None
    probs: Optional[torch.FloatTensor] = None
    last_hidden_state: Optional[Tuple[torch.FloatTensor]] = None
    contacts: Optional[torch.FloatTensor] = None
    all_hidden_states: Optional[Tuple[torch.FloatTensor]] = None


def symmetrize(x):
    """
    Symmetrize a tensor along its last two dimensions.
    
    Typically used for attention/contact maps to ensure mutual interactions are symmetric.

    Args:
        x (torch.Tensor): Input tensor with shape [..., N, N].

    Returns:
        torch.Tensor: Symmetrized tensor.
    """
    return x + x.transpose(-1, -2)


class gLMMultiHead(nn.Module):
    """
    Roberta head for masked language modeling with multiple prediction heads.

    Each token is projected to multiple predictions, useful for structured output tasks.
    """

    def __init__(self, config):
        super().__init__()
        self.num_pred = config.num_pred  # Number of prediction heads per token.
        self.num_pc = config.num_pc      # Output dimensionality per prediction.
        self.dense = nn.Linear(config.hidden_size, config.num_pc * self.num_pred)  # Main output layer.
        self.hidden_size = config.hidden_size
        self.predict_probs = config.predict_probs  # Flag to indicate if probability prediction is used.

        if config.predict_probs:
            self.dense_prob = nn.Linear(config.hidden_size, self.num_pred)  # Predicts probabilities per token.

    def forward(self, features, **kwargs):
        """
        Forward pass for multi-head prediction.

        Args:
            features (torch.Tensor): Input embeddings of shape [batch, seq_len, hidden_size].

        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor]]: 
                - Output tensor reshaped to [batch, seq_len, num_pred, num_pc].
                - Optional probabilities per token if enabled.
        """
        x = self.dense(features)
        x_shape = list(x.shape)
        x = x.view(*x_shape[:-1], self.num_pred, self.num_pc)

        if self.predict_probs:
            probs = self.dense_prob(features)
        else:
            probs = None

        return x, probs


class gLMHead(nn.Module):
    """
    Roberta head for standard masked language modeling.

    A single dense layer that projects the hidden representation to the output size.
    """

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.num_pc)

    def forward(self, features, **kwargs):
        """
        Forward pass for the LM head.

        Args:
            features (torch.Tensor): Input embeddings of shape [batch, seq_len, hidden_size].

        Returns:
            torch.Tensor: Output logits of shape [batch, seq_len, num_pc].
        """
        x = self.dense(features)
        return x


class ContactPredictionHead(nn.Module):
    """
    Contact prediction head adapted from fair-esm.

    Uses symmetrized attention maps to compute contact probabilities using a linear regression layer.
    """

    def __init__(
        self,
        in_features: int,
        bias=True,
    ):
        super().__init__()
        self.in_features = in_features  # Input dimensionality (typically layers × heads).
        self.regression = nn.Linear(in_features, 1, bias)  # Linear regression for each attention pair.

    def forward(self, tokens, attentions):
        """
        Forward pass for contact prediction.

        Args:
            tokens (torch.Tensor): Not used directly, placeholder for API compatibility.
            attentions (torch.Tensor): Attention weights of shape [batch, layers, heads, seq_len, seq_len].

        Returns:
            torch.Tensor: Contact map features of shape [batch, seq_len, seq_len, channels].
        """
        batch_size, layers, heads, seqlen, _ = attentions.size()
        # Reshape to [batch, layers*heads, seq_len, seq_len]
        attentions = attentions.view(batch_size, layers * heads, seqlen, seqlen)

        # Move attentions to same device as regression weights
        attentions = attentions.to(self.regression.weight.device)

        # Symmetrize across the final two dimensions
        attentions = symmetrize(attentions)

        # Rearrange to [batch, seq_len, seq_len, channels]
        attentions = attentions.permute(0, 2, 3, 1)

        return attentions

class gLM_base(RobertaModel):
    """
    Base class extending RobertaModel for masked language modeling with optional contact prediction and 
    multiple prediction heads.

    Attributes:
        lm_head (nn.Module): Prediction head for token-level output, can be single or multi-headed.
        contact_head (nn.Module): Head for predicting contact maps from attention weights.
    """

    _keys_to_ignore_on_save = [r"lm_head.decoder.weight", r"lm_head.decoder.bias"]
    _keys_to_ignore_on_load_missing = [r"position_ids", r"lm_head.decoder.weight", r"lm_head.decoder.bias"]
    _keys_to_ignore_on_load_unexpected = [r"pooler"]

    def __init__(self, config):
        super().__init__(config)
        self.roberta = RobertaModel(config, add_pooling_layer=False)
        self.num_pred = config.num_pred
        self.predict_probs = config.predict_probs

        # Initialize appropriate LM head based on the number of predictions
        if self.num_pred == 1:
            self.lm_head = gLMHead(config)
        else:
            self.lm_head = gLMMultiHead(config)

        # Contact prediction head based on number of layers × heads
        self.contact_head = ContactPredictionHead(config.num_hidden_layers * config.num_attention_heads)

        # Remove tied weight keys from ignore list if not needed
        self.update_keys_to_ignore(config, ["lm_head.decoder.weight"])

        # Standard post-initialization from Transformers
        self.post_init()

    def update_keys_to_ignore(self, config, del_keys_to_ignore):
        """
        Updates the keys to ignore during weight save/load depending on model configuration.

        Args:
            config (PretrainedConfig): Model configuration.
            del_keys_to_ignore (List[str]): Keys to be removed from ignore list.
        """
        if not config.tie_word_embeddings:
            # Clone lists to avoid modifying class variables
            self._keys_to_ignore_on_save = [k for k in self._keys_to_ignore_on_save if k not in del_keys_to_ignore]
            self._keys_to_ignore_on_load_missing = [
                k for k in self._keys_to_ignore_on_load_missing if k not in del_keys_to_ignore
            ]

    def get_output_embeddings(self):
        """Returns the decoder head, used for tied weight retrieval."""
        return self.lm_head.decoder

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        masked_tokens: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], MultiPredMaskedLMOutput]:
        """
        Forward pass for masked language modeling with optional contact prediction.

        Returns:
            MultiPredMaskedLMOutput or tuple: Includes predictions, optional probabilities, 
            hidden states, and contact map.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]  # Final hidden states from Roberta encoder

        predicted_probs = None
        if self.num_pred > 1:
            # Multi-head prediction output
            predicted_embeds, predicted_probs = self.lm_head(sequence_output)
        else:
            predicted_embeds = self.lm_head(sequence_output)

        total_loss = None
        if not return_dict:
            output = (predicted_embeds,) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output

        # Compute contact predictions if attention outputs are enabled
        contacts = None
        if output_attentions:
            attn_weights = []
            for attn_per_layer in outputs['attentions']:
                attn_weights.append(attn_per_layer)
            attentions = torch.stack(attn_weights, 1)
            contacts = self.contact_head(inputs_embeds, attentions)

        # Collect all hidden states if requested
        all_hidden_states = []
        if output_hidden_states:
            for states_per_layer in outputs['hidden_states']:
                all_hidden_states.append(states_per_layer)
            all_hidden_states = torch.stack(all_hidden_states, 1)

        return MultiPredMaskedLMOutput(
            logits_all_preds=predicted_embeds,
            probs=predicted_probs,
            last_hidden_state=sequence_output,
            contacts=contacts,
            all_hidden_states=all_hidden_states,
        )


class gLM(RobertaModel):
    """
    Top-level model for gLM, wrapping gLM_base with an optional embedding projection layer.

    Attributes:
        roberta (gLM_base): The main model performing token-level prediction.
        dense (nn.Linear): Projection layer to map from `emb_dim` to `hidden_size`.
    """

    def __init__(self, config):
        super().__init__(config)
        self.roberta = gLM_base(config)
        self.dense = nn.Linear(config.emb_dim, config.hidden_size)  # Project embeddings if needed
        self.output_attentions = config.output_attentions

        self.post_init()  # Standard Transformer model initialization

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        masked_tokens: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs
    ) -> Union[Tuple[torch.Tensor], MultiPredMaskedLMOutput]:
        """
        Forward pass for gLM model. Applies embedding projection and passes to base model.

        Returns:
            MultiPredMaskedLMOutput or tuple: Output containing logits, contact map, hidden states, etc.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Project input embeddings to match hidden size
        inputs_embeds = self.dense(inputs_embeds)

        # Pass through the base model
        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=self.output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            labels=labels,
            masked_tokens=masked_tokens,
        )

        return outputs