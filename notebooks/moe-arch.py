import torch, os, evaluate, sys
import torch.nn as nn
import pandas as pd
import dataclasses, sys, pickle, json
import torch.nn.functional as F
from typing import List, Optional
import matplotlib.pyplot as plt, random, numpy as np
from datasets import load_dataset,Dataset,DatasetDict
from transformers import RobertaModel, RobertaConfig
from transformers import DataCollatorWithPadding,AutoModelForSequenceClassification, Trainer, TrainingArguments,AutoTokenizer,AutoModel,AutoConfig
from transformers.modeling_outputs import TokenClassifierOutput
from torch.utils.data import DataLoader
from transformers import AdamW,get_scheduler
from datasets import load_metric
from tqdm.auto import tqdm
from torch.nn import DataParallel
from transformers import RobertaTokenizer
from torch.utils.data import Dataset
import torch

sys.path.append("/home/pritam.k/research/data-moe")

import torch
import torch.nn as nn
import torch.nn.functional as F

class MoE(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_experts=4, num_difficulties=3, capacity_factor=1.5):
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

    def forward(self,x,difficulty_labels):
        batch_size, input_dim = x.size()
        difficulty_embeds = self.difficulty_embedding(difficulty_labels)
        
        gate_input = torch.cat([x, difficulty_embeds], dim=1)  # (batch_size, 2 * input_dim)

        # Compute gate logits
        gate_logits = self.gate(gate_input)  # (batch_size, num_experts)
        
        topk = 2
        weights, topk_indices = gate_logits.topk(topk, dim=1)
        topk_probs = F.softmax(weights, dim=1)  # Corrected variable name

        # Initialize output
        output = torch.zeros(batch_size, self.experts[0].out_features, device=x.device)

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
            expert_mask = (topk_indices[mask] == i)  # (num_selected, topk)
            weight = topk_probs[mask][expert_mask]  # (num_selected * occurrences,)

            # Sum weights per instance
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
            print(layer)
            xx=hidden_states[:, 0, :]
            print(layer.intermediate)
            moe_output, routing_info = layer.intermediate(x=xx,difficulty_labels=difficulties)  # Using [CLS] token

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

model = RobertaWithMoE(
    model_name='roberta-base',
    num_experts=4,
    hidden_dim=3072,
    num_difficulties=3,
    num_labels=2
)
print(model)


# class CustomDataset(torch.utils.data.Dataset):
#     def __init__(self, texts, labels, difficulties, tokenizer, max_length=128):
#         self.texts = texts
#         self.labels = labels
#         self.difficulties = difficulties
#         self.tokenizer = tokenizer
#         self.max_length = max_length
        
#         assert len(self.texts) == len(self.labels) == len(self.difficulties), "Dataset length mismatch!"

#     def __len__(self):
#         return len(self.texts)  # This should return the size of your dataset

#     def __getitem__(self, idx):
#         if idx >= len(self.texts):
#             raise IndexError(f"Index {idx} out of bounds for dataset size {len(self.texts)}")
        
#         text = self.texts[idx]
#         label = self.labels[idx]
#         difficulty = self.difficulties[idx]
        
#         encoding = self.tokenizer.encode_plus(
#             text,
#             add_special_tokens=True,
#             max_length=self.max_length,
#             padding='max_length',
#             truncation=True,
#             return_attention_mask=True,
#             return_tensors='pt',
#         )
        
#         return {
#             'input_ids': encoding['input_ids'].flatten(),
#             'attention_mask': encoding['attention_mask'].flatten(),
#             'labels': torch.tensor(label, dtype=torch.long),
#             'difficulties': torch.tensor(difficulty, dtype=torch.long)
#         }
# # Example data
# texts = [
#     "I love this product!",
#     "This product was terrible.",
#     "It's okay, nothing special.",
#     "Amazing service, highly recommend.",
#     "Not my cup of tea.",
#     "The experience was ambiguous and unclear.",
#     "The functionality is hard to use.",
#     "Could be better.",
#     "Outstanding performance!",
#     "Mediocre at best."
# ]
# labels = [1, 0, 0, 1, 0, 1, 0, 0, 1, 0]  # 1: Positive, 0: Negative
# difficulties = [0, 0, 0, 0, 0, 1, 2, 1, 0, 1]  # 0: easy, 1: ambiguous, 2: hard

# from torch.utils.data import DataLoader
# import torch.optim as optim
# from collections import defaultdict
# import matplotlib.pyplot as plt

# # Initialize tokenizer
# tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

# # Create Dataset and DataLoader
# dataset = CustomDataset(texts, labels, difficulties, tokenizer)
# dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

# # Initialize Model
# model = RobertaWithMoE(
#     model_name='roberta-base',
#     num_experts=4,
#     hidden_dim=3072,
#     num_difficulties=3,
#     num_labels=2
# )
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# model = model.to(device)

# # Define Optimizer
# optimizer = optim.AdamW(model.parameters(), lr=2e-5)

# # Initialize Expert Usage Counters
# expert_usage = defaultdict(lambda: defaultdict(int))  # expert_usage[difficulty][expert] = count



# # Training Loop
# epochs = 3
# model.train()
# for epoch in range(epochs):
#     print(f"Epoch {epoch + 1}/{epochs}")
#     for batch in dataloader:
#         input_ids = batch['input_ids'].to(device)
#         attention_mask = batch['attention_mask'].to(device)
#         labels = batch['labels'].to(device)
#         difficulties = batch['difficulties'].to(device)
#         optimizer.zero_grad()
#         (loss, logits), routing_info = model(
#             input_ids=input_ids,
#             attention_mask=attention_mask,
#             labels=labels,
#             difficulties=difficulties
#         )
#         loss.backward()
#         optimizer.step()

#         # Collect routing info
#         # routing_info shape: (batch_size, topk)
#         routing_info = routing_info.cpu().numpy()
#         difficulties_np = difficulties.cpu().numpy()

#         for d, experts in zip(difficulties_np, routing_info):
#             for e in experts:
#                 expert_usage[d][e] += 1

#         print(f"Loss: {loss.item()}")

#     # Print expert usage after each epoch
#     print(f"Expert usage after epoch {epoch + 1}:")
#     for d in range(3):
#         print(f"  Difficulty {d}:")
#         for e in range(model.roberta.encoder.layer[0].intermediate.num_experts):
#             print(f"    Expert {e}: {expert_usage[d][e]}")
#     print()
