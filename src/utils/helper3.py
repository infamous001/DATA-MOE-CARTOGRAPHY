import dataclasses
from typing import List

import torch
import torch.nn.functional as F
from simple_parsing.helpers import Serializable
from dataclasses import dataclass
from torch import nn
from transformers.modeling_outputs import TokenClassifierOutput
from transformers import DataCollatorWithPadding,AutoModelForSequenceClassification, Trainer, TrainingArguments,AutoTokenizer,AutoModel,AutoConfig

class ConfiguredMetric:
    def __init__(self, metric, *metric_args, **metric_kwargs):
        self.metric = metric
        self.metric_args = metric_args
        self.metric_kwargs = metric_kwargs
    
    def add(self, *args, **kwargs):
        return self.metric.add(*args, **kwargs)
    
    def add_batch(self, *args, **kwargs):
        return self.metric.add_batch(*args, **kwargs)

    def compute(self, *args, **kwargs):
        return self.metric.compute(*args, *self.metric_args, **kwargs, **self.metric_kwargs)

    @property
    def name(self):
        return self.metric.name

    def _feature_names(self):
        return self.metric._feature_names()



def z_loss(logits):
    log_sum_exp=torch.logsumexp(logits,dim=1)
    sq_log_sum_exp=log_sum_exp=log_sum_exp**2
    loss=torch.mean(sq_log_sum_exp)
    return loss

def b_loss(logits,alpha=1e-2):
    T,N=logits.shape
    probs=torch.softmax(logits,dim=1)

    argmax_experts=torch.argmax(probs,dim=1)
    f=torch.zeros(N,device=logits.device)
    for i in range(N):
        f[i]=(argmax_experts==i).float().mean()
    P=probs.mean(dim=0)

    loss=alpha*N*torch.sum(f*P)
    return loss

@dataclasses.dataclass
class MoeArgs:
  def __init__(self, num_experts: int,num_experts_per_tok):
    self.num_experts=num_experts
    self.num_experts_per_tok=num_experts_per_tok

class MoeLayer(nn.Module):
    def __init__(self, experts: List[nn.Module], gate: nn.Module, moe_args: MoeArgs):
        super().__init__()
        assert len(experts) > 0
        self.gate = gate
        self.experts = nn.ModuleList(experts)
        self.args = moe_args
        #self.expert_assignments = None

    def forward(self, inputs: torch.Tensor):
        gate_logits = self.gate(inputs)
        loss_z=z_loss(gate_logits)
        loss_b=b_loss(gate_logits)
        weights, selected_experts = torch.topk(gate_logits, self.args.num_experts_per_tok)
        weights = F.softmax(weights, dim=1, dtype=torch.float).to(inputs.dtype)
        results = torch.zeros(inputs.size(0),9).to("cuda:0")
        expert_tracking = {i: [] for i in range(self.args.num_experts)}
        for current_expert_index, current_expert in enumerate(self.experts):
            token_index, token_expert_index = torch.where(selected_experts == current_expert_index)
            for idx in token_index.cpu().numpy():
                expert_tracking[current_expert_index].append(idx)
            results[token_index] = current_expert(inputs[token_index])
        return results, loss_z, loss_b, expert_tracking


class ModelArgs:
    def __init__(self, dim: int, hidden_dim: int, num_labels: int, moe: MoeArgs):
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.num_labels = num_labels
        self.moe = moe


class FeedForward(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        self.w1 = nn.Linear(args.dim, args.hidden_dim, bias=False)
        self.w2 = nn.Linear(args.hidden_dim, args.dim, bias=False)
        self.w3 = nn.Linear(args.dim, args.hidden_dim, bias=False)
        self.classifier=nn.Linear(768,args.moe.num_experts*args.num_labels)

    def forward(self, x) -> torch.Tensor:
        a=self.w2(nn.functional.silu(self.w1(x)) * self.w3(x))
        return self.classifier(a)


class CustomModel(nn.Module):
  def __init__(self,checkpoint,args: ModelArgs):
    super(CustomModel,self).__init__()

    self.model = model = AutoModel.from_pretrained(checkpoint,config=AutoConfig.from_pretrained(checkpoint, output_attentions=True,output_hidden_states=True))
    # for param in self.model.parameters():
    #     param.requires_grad = False
    self.args=args
    self.moe_layer=MoeLayer(experts=[FeedForward(args=args) for _ in range(args.moe.num_experts)],gate=nn.Linear(args.dim, args.moe.num_experts, bias=False),moe_args=args.moe,)
    self.classifier = nn.Linear(768,args.moe.num_experts*args.num_labels)

  def forward(self, input_ids=None, attention_mask=None,labels=None):
    outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
    logits,loss_z,loss_b,expert_tracking = self.moe_layer(outputs[0][:,0,:].view(-1,768))
    #expert_assignments = self.moe_layer.expert_assignments
    #print(expert_tracking)
    # logits=self.classifier(moe_outputs)

    loss = None
    if labels is not None:
      loss_fct = nn.CrossEntropyLoss()
      loss = loss_fct(logits.view(-1, self.args.moe.num_experts*self.args.num_labels), labels.view(-1))
      loss=loss+loss_b+loss_z

    return TokenClassifierOutput(loss=loss, logits=logits, hidden_states=outputs.hidden_states,attentions=outputs.attentions), expert_tracking

        #return outputs