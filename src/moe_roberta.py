import torch, os, evaluate, sys
import torch.nn as nn
import pandas as pd

import dataclasses, sys, pickle, json
import torch.nn.functional as F

from typing import List, Optional

import matplotlib.pyplot as plt, random, numpy as np

from datasets import load_dataset,Dataset,DatasetDict
from transformers import DataCollatorWithPadding,AutoModelForSequenceClassification, Trainer, TrainingArguments,AutoTokenizer,AutoModel,AutoConfig
from transformers.modeling_outputs import TokenClassifierOutput
from torch.utils.data import DataLoader
from transformers import AdamW,get_scheduler
from datasets import load_metric
from tqdm.auto import tqdm

from torch.nn import DataParallel

sys.path.append("/home/pritam.k/research/data-moe")

from transformers import RobertaModel, RobertaConfig
# from src.utils.helper import CustomModel, ModelArgs, MoeArgs
from src.utils.helper import ConfiguredMetric


class MoE(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_experts=4, num_difficulties=3, capacity_factor=1.5):
        """
        Mixture of Experts (MoE) Layer with Difficulty-Aware Gating.

        Args:
            input_dim (int): Input feature dimension.
            hidden_dim (int): Hidden feature dimension for each expert.
            num_experts (int): Number of experts.
            num_difficulties (int): Number of difficulty categories.
            capacity_factor (float): Determines the capacity per instance.
        """
        super(MoE, self).__init__()
        self.num_experts = num_experts
        self.num_difficulties = num_difficulties
        self.capacity = int(capacity_factor)

        # Define experts
        self.experts = nn.ModuleList([nn.Linear(input_dim, hidden_dim) for _ in range(num_experts)])

        # Define difficulty embeddings
        self.difficulty_embedding = nn.Embedding(num_difficulties, input_dim)

        # Define gating network (conditioned on input and difficulty)
        self.gate = nn.Linear(input_dim * 2, num_experts)

    def forward(self, x, difficulty_labels=None):
        """
        Forward pass for MoE layer.

        Args:
            x (Tensor): Input tensor of shape (batch_size, input_dim).
            difficulty_labels (Tensor): Tensor of difficulty labels of shape (batch_size,).

        Returns:
            Tensor: Output tensor of shape (batch_size, hidden_dim).
            Tensor: Routing information tensor of shape (batch_size, topk).
        """
        print(x.shape)
        # batch_size = x.size(0)
        # seq_length = x.size(1)
        # hidden_dim = x.size(2)
        #print(difficulty_labels)
        batch_size, input_dim, _ = x.size()

        # Get difficulty embeddings
        difficulty_embeds = self.difficulty_embedding(difficulty_labels)  # (batch_size, input_dim)

        # Concatenate input with difficulty embeddings
        gate_input = torch.cat([x, difficulty_embeds], dim=1)  # (batch_size, 2 * input_dim)

        # Compute gate logits
        gate_logits = self.gate(gate_input)  # (batch_size, num_experts)
        gate_probs = F.softmax(gate_logits, dim=1)  # (batch_size, num_experts)

        # Select top-k experts
        topk = 2  # Number of experts to select
        topk_probs, topk_indices = gate_probs.topk(topk, dim=1)  # Each row has top-k probabilities and indices

        # Initialize output
        output = torch.zeros(batch_size, self.experts[0].out_features).to(x.device)

        # Routing info for analysis
        routing_info = topk_indices  # (batch_size, topk)

        # Iterate over experts
        for i in range(self.num_experts):
            # Find instances assigned to expert i
            mask = (topk_indices == i).any(dim=1)  # (batch_size,)
            if mask.sum() == 0:
                continue
            selected_instances = x[mask]  # (num_selected, input_dim)
            selected_probs = topk_probs[mask]  # (num_selected, topk)

            # Pass through expert
            expert_output = self.experts[i](selected_instances)  # (num_selected, hidden_dim)

            # Weight by gate probabilities
            # Sum the probabilities for expert i (since it can be selected in top-k)
            # Find where expert i was selected in top-k
            expert_mask = (topk_indices[mask] == i)  # (num_selected, topk)
            weight = topk_probs[mask][expert_mask]  # (num_selected * occurrences,)

            # Sum weights per instance (if multiple occurrences, sum them)
            # Assuming top-k=2, and each expert can be selected at most once per instance
            # So, weight.sum(dim=1) is equivalent to weight.squeeze(1) if topk=2
            weight = expert_mask.float() * topk_probs[mask]  # (num_selected, topk)
            weight = weight.sum(dim=1, keepdim=True)  # (num_selected, 1)

            output[mask] += expert_output * weight  # Weighted sum

        return output, routing_info




class RobertaWithMoE(nn.Module):
    def __init__(self, model_name='roberta-base', num_experts=4, hidden_dim=3072, num_difficulties=3, num_labels=2):
        """
        Custom RoBERTa model integrated with MoE layers.

        Args:
            model_name (str): Pre-trained RoBERTa model name.
            num_experts (int): Number of experts in MoE layers.
            hidden_dim (int): Hidden dimension for each expert.
            num_difficulties (int): Number of difficulty categories.
            num_labels (int): Number of classification labels.
        """
        super(RobertaWithMoE, self).__init__()
        self.roberta = RobertaModel.from_pretrained(model_name)
        config = self.roberta.config

        # Replace the intermediate dense layer in each Transformer block with MoE
        for layer in self.roberta.encoder.layer:
            # Original intermediate layer
            original_fc = layer.intermediate.dense

            # New MoE layer
            moe_layer = MoE(
                input_dim=original_fc.in_features,
                hidden_dim=hidden_dim,
                num_experts=num_experts,
                num_difficulties=num_difficulties
            )
            layer.intermediate = moe_layer

        # Classification head
        self.classifier = nn.Linear(config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask=None, labels=None, difficulties=None):
        """
        Forward pass for the custom RoBERTa model with MoE.

        Args:
            input_ids (Tensor): Input token IDs.
            attention_mask (Tensor): Attention masks.
            labels (Tensor): Labels for classification.
            difficulties (Tensor): Difficulty labels.

        Returns:
            Tuple: (loss, logits) if labels are provided; otherwise, (logits, routing_info).
        """
        # Get RoBERTa outputs
        outputs = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        hidden_states = outputs.last_hidden_state  # (batch_size, seq_length, hidden_size)

        # Ensure difficulties are passed correctly
        if difficulties is None:
            raise ValueError("Difficulties tensor must be provided.")

        # Pass through transformer layers with MoE
        for layer in self.roberta.encoder.layer:
            moe_output, routing_info = layer.intermediate(hidden_states[:, 0, :], difficulties)  # Using [CLS] token

            # Continue with the output dense layer and other components
            intermediate_output = moe_output  # (batch_size, hidden_dim)
            layer_output = layer.output.dense(intermediate_output)  # (batch_size, hidden_size)
            layer_output = layer.output.dropout(layer_output)
            layer_output = layer.output.LayerNorm(layer_output + outputs.last_hidden_state[:, 0, :])  # Residual connection
            hidden_states[:, 0, :] = layer_output  # Update [CLS] token

        # Pooling (use [CLS] token representation)
        pooled_output = hidden_states[:, 0, :]  # (batch_size, hidden_size)

        # Classification
        logits = self.classifier(pooled_output)  # (batch_size, num_labels)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.classifier.out_features), labels.view(-1))

        return (loss, logits) if loss is not None else (logits, routing_info)