
# %% [code] {"execution":{"iopub.status.busy":"2025-06-24T16:18:03.374384Z","iopub.execute_input":"2025-06-24T16:18:03.374683Z","iopub.status.idle":"2025-06-24T16:18:09.106873Z","shell.execute_reply.started":"2025-06-24T16:18:03.374649Z","shell.execute_reply":"2025-06-24T16:18:09.106320Z"}}
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer
import pytorch_lightning as pl
import numpy as np
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint


# %% [code] {"execution":{"iopub.status.busy":"2025-06-24T16:18:09.107748Z","iopub.execute_input":"2025-06-24T16:18:09.108270Z","iopub.status.idle":"2025-06-24T16:18:10.572053Z","shell.execute_reply.started":"2025-06-24T16:18:09.108221Z","shell.execute_reply":"2025-06-24T16:18:10.571206Z"}}
tokenizer = AutoTokenizer.from_pretrained("gpt2")  # Simple subword tokenizer
tokenizer.pad_token = tokenizer.eos_token
def tokenize(example):
    return tokenizer(example['text'], truncation=True, padding="max_length", max_length=1024)

dataset = load_dataset("roneneldan/TinyStories", split="train[:10%]")  # small subset for demo
dataset = dataset.map(tokenize, batched=True)
dataset.set_format(type="torch", columns=["input_ids"])

# Prepare input-output pairs
class LanguageModelingDataset(torch.utils.data.Dataset):
    def __init__(self, tokenized_dataset):
        self.data = tokenized_dataset["input_ids"]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        y = x.clone()
        return x, y

train_ds = LanguageModelingDataset(dataset)
train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)


# %% [code] {"execution":{"iopub.status.busy":"2025-06-24T16:18:10.572949Z","iopub.execute_input":"2025-06-24T16:18:10.573247Z","iopub.status.idle":"2025-06-24T16:18:10.583587Z","shell.execute_reply.started":"2025-06-24T16:18:10.573216Z","shell.execute_reply":"2025-06-24T16:18:10.582746Z"}}
class GatedSSMBlock(nn.Module):
    def __init__(self, d_model, dropout=0.2):
        super().__init__()
        self.U = nn.Linear(d_model, d_model)
        self.F = nn.Linear(d_model, d_model)
        self.O = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        forget_gate = torch.sigmoid(self.F(x))
        update = torch.tanh(self.U(x))
        output_gate = torch.sigmoid(self.O(x))
        h = forget_gate * x + (1 - forget_gate) * update
        return self.dropout(self.norm(output_gate * h))

class CrossHeadRouter(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.router = nn.Linear(d_model * num_heads, d_model)

    def forward(self, head_outputs):
        x = torch.cat(head_outputs, dim=-1)
        return self.router(x)

class TokenWiseFeedForward(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
        )

    def forward(self, x):
        return self.ff(x)

class ParallelSSMHeads(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.heads = nn.ModuleList([GatedSSMBlock(d_model) for _ in range(num_heads)])
        self.router = CrossHeadRouter(d_model, num_heads)
        self.tokenwise_ffn = TokenWiseFeedForward(d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        head_outputs = [head(x) for head in self.heads]
        fused = self.router(head_outputs)
        fused = fused + self.tokenwise_ffn(fused)
        return self.norm(fused)

class HybridBlock(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.parallel_ssm = ParallelSSMHeads(d_model, num_heads)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        x_ssm = self.parallel_ssm(x)
        return self.norm(x + x_ssm)


# %% [code] {"execution":{"iopub.status.busy":"2025-06-24T16:18:10.584490Z","iopub.execute_input":"2025-06-24T16:18:10.584734Z","iopub.status.idle":"2025-06-24T16:18:10.599557Z","shell.execute_reply.started":"2025-06-24T16:18:10.584717Z","shell.execute_reply":"2025-06-24T16:18:10.598910Z"}}
class LazySSMLanguageModel(pl.LightningModule):
    def __init__(self, vocab_size, d_model=768, depth=2, num_heads=3, max_seq_len=1024, lr=1e-4):
        super().__init__()
        self.save_hyperparameters()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Parameter(torch.zeros(1, max_seq_len, d_model))
        self.blocks = nn.Sequential(*[HybridBlock(d_model, num_heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, idx):
        B, T = idx.size()
        tok_emb = self.token_embedding(idx)
        pos_emb = self.pos_embedding[:, :T, :]
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.norm(x)
        return self.head(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))

    # Penalize repetition: encourage diversity
        preds = torch.argmax(logits, dim=-1)
        repeat_penalty = (preds[:, 1:] == preds[:, :-1]).float().mean()
        loss += 10.0 * repeat_penalty  # 🔁 Heavy weight on repetition

        self.log("train_loss", loss)
        return loss


    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=0.05)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=self.hparams.lr,
            total_steps=5000, pct_start=0.1, anneal_strategy='cos',
        )
        return [optimizer], [scheduler]


    def generate(self, idx, max_new_tokens=100, repetition_penalty=5.0):
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.hparams.max_seq_len:]

            logits = self(idx_cond)
            logits = logits[:, -1, :]

        # Apply repetition penalty
            for b in range(idx.size(0)):
                for token in set(idx[b].tolist()):
                    logits[b, token] /= repetition_penalty

            probs = torch.softmax(logits, dim=-1)
            top_k = 25
            topk_probs, topk_indices = torch.topk(probs, top_k)
            next_token = topk_indices.gather(1, torch.multinomial(topk_probs, 1))
            idx = torch.cat([idx, next_token], dim=1)
        return idx



# %% [code] {"execution":{"iopub.status.busy":"2025-06-24T16:19:44.156019Z","iopub.execute_input":"2025-06-24T16:19:44.156350Z","iopub.status.idle":"2025-06-24T16:19:44.985452Z","shell.execute_reply.started":"2025-06-24T16:19:44.156326Z","shell.execute_reply":"2025-06-24T16:19:44.984527Z"}}

if __name__ == "__main__":
    vocab_size = tokenizer.vocab_size
    model = LazySSMLanguageModel(vocab_size=vocab_size)

    trainer = Trainer(
        max_epochs=5,
        accelerator="gpu",
        devices=2,
        precision="16-mixed",
        strategy="ddp",  # Distributed data parallel for 2 T4s
        callbacks=[ModelCheckpoint(monitor="train_loss")]
    )

    trainer.fit(model, train_loader)


# %% [code] {"execution":{"iopub.status.busy":"2025-06-24T16:18:11.484121Z","iopub.status.idle":"2025-06-24T16:18:11.484386Z","shell.execute_reply.started":"2025-06-24T16:18:11.484234Z","shell.execute_reply":"2025-06-24T16:18:11.484244Z"}}
context = "The arabs steam through the golden horn, about to besiege constantinople. Emperor Leo 3 engages the arab fleet with cunning tactics, including:"
tokens = tokenizer(context, return_tensors="pt")["input_ids"].to(model.device)
with torch.no_grad():
    gen_tokens = model.generate(tokens, max_new_tokens=100)
print(tokenizer.decode(gen_tokens[0], skip_special_tokens=True))

