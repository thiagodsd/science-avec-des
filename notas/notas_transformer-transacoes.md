# Transformer sobre transações bancárias

Trasnformer treinado pra prever o próximo evento da vida de um cliente. 
Cada evento vira um token; o histórico é a sequência.

**Referências:**
+ https://github.com/karpathy/nanoGPT/blob/master/model.py
+ https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf
+ https://arxiv.org/abs/1801.10198
+ https://arxiv.org/abs/1706.03762
+ https://jalammar.github.io/illustrated-gpt2/
+ https://arxiv.org/abs/2104.09864.


```python
from datetime import date
from pathlib import Path

import kagglehub
import numpy as np
import polars as pl
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset

CTX, BS, LR, EPOCHS, WARMUP, N_BINS = 512, 64, 1e-3, 30, 1, 6
device = "cuda" if torch.cuda.is_available() else "cpu"
```

## 1. Eventos

Um evento por linha:

$$
\begin{aligned}
\texttt{client\_id} &\to \text{conta} \\
\texttt{ts} &\to \text{data} \\
\texttt{event\_type} &\to \text{tipo} : \text{operação} : \text{categoria} \\
\texttt{amount} &\to \text{valor}
\end{aligned}
$$

Três fontes, de https://www.kaggle.com/datasets/marceloventura/the-berka-dataset:
+ transações
+ empréstimos
+ cartões

Empréstimo e cartão entram como evento com `event_type` próprio; cartão tem `amount = 0`.

```python
raw = Path(kagglehub.dataset_download("marceloventura/the-berka-dataset"))

rd = lambda f: pl.read_csv(raw / f, separator=";", infer_schema_length=0)
ymd = lambda c: pl.col(c).str.slice(0, 6).str.strptime(pl.Date, "%y%m%d")
clean = lambda c: pl.col(c).fill_null("").str.strip_chars()

trans = rd("trans.csv").select(
    pl.col("account_id").alias("client_id"),
    ymd("date").alias("ts"),
    pl.concat_str([clean("type"), clean("operation"), clean("k_symbol")], separator=":")
    .str.replace_all(r":+$", "")
    .str.replace_all(r"::", ":")
    .alias("event_type"),
    pl.col("amount").cast(pl.Float64),
    pl.col("trans_id").cast(pl.Int64).alias("order"),
)
loan = rd("loan.csv").select(
    pl.col("account_id").alias("client_id"),
    ymd("date").alias("ts"),
    pl.concat_str([pl.lit("LOAN:"), pl.col("duration")]).alias("event_type"),
    pl.col("amount").cast(pl.Float64),
    pl.lit(-1, dtype=pl.Int64).alias("order"),
)
card = rd("card.csv").join(rd("disp.csv").select("disp_id", "account_id"), on="disp_id").select(
    pl.col("account_id").alias("client_id"),
    ymd("issued").alias("ts"),
    pl.concat_str([pl.lit("CARD:"), pl.col("type")]).alias("event_type"),
    pl.lit(0.0).alias("amount"),
    pl.lit(-2, dtype=pl.Int64).alias("order"),
)
ev = pl.concat([trans, loan, card]).sort(["client_id", "ts", "order"]).drop("order")

val_start, test_start = date(1997, 7, 1), date(1998, 1, 1)
```

Tabela `ev`, com zoom dado em um cliente (tipos em tcheco: PRIJEM crédito, VYDAJ débito, VKLAD depósito, PREVOD Z UCTU transferência, UROK juros, VYBER saque):

| client_id | ts         | event_type           | amount  |
|-----------|------------|----------------------|--------:|
| 2         | 1993-02-26 | PRIJEM:VKLAD         |  1100.0 |
| 2         | 1993-03-12 | PRIJEM:PREVOD Z UCTU | 20236.0 |
| 2         | 1993-03-28 | PRIJEM:VKLAD         |  3700.0 |
| 2         | 1993-03-31 | PRIJEM:UROK          |    13.5 |
| 2         | 1993-04-12 | PRIJEM:PREVOD Z UCTU | 20236.0 |
| 2         | 1993-04-27 | VYDAJ:VYBER          | 11000.0 |
| 2         | 1993-04-30 | PRIJEM:UROK          |   109.5 |
| 2         | 1993-05-12 | PRIJEM:PREVOD Z UCTU | 20236.0 |

## 2. Tokenização

Cada evento vira dois tokens: 
+ `dt|<bucket>` (dias desde o evento anterior)
+ `<event_type>|<faixa>` (faixa de valor por quantil, por tipo de evento). Faixas ajustadas só no treino. 

Sequência de um cliente: `<bos> dt s dt s ...`.

```python
PAD, BOS, UNK = 0, 1, 2
DT_EDGES = [0, 1, 2, 3, 5, 7, 10, 14, 21, 30, 45, 60, 90, 180, 365]


def dt_bucket(days):
    return int(np.searchsorted(DT_EDGES, days, side="right")) - 1


class Vocab:
    def __init__(self, tokens, amount_edges):
        self.itos = ["<pad>", "<bos>", "<unk>"] + sorted(set(tokens))
        self.stoi = {t: i for i, t in enumerate(self.itos)}
        self.amount_edges = amount_edges
        self.state_ids = np.array([i for i, t in enumerate(self.itos) if i > 2 and not t.startswith("dt|")])
        self.dt_ids = np.array([i for i, t in enumerate(self.itos) if t.startswith("dt|")])

    def __len__(self):
        return len(self.itos)

    def amount_bin(self, etype, amount):
        edges = self.amount_edges.get(etype, [])
        return int(np.searchsorted(edges, abs(amount), side="right")) if len(edges) else 0

    def state(self, etype, amount):
        return self.stoi.get(f"{etype}|{self.amount_bin(etype, amount)}", UNK)

    def dt(self, days):
        return self.stoi[f"dt|{dt_bucket(days)}"]


def fit_vocab(events, n_bins=N_BINS):
    edges = {}
    for (etype,), grp in events.group_by("event_type"):
        a = np.abs(grp["amount"].to_numpy())
        a = a[a > 0]
        if len(np.unique(a)) < n_bins:
            edges[etype] = []
            continue
        edges[etype] = np.unique(np.quantile(a, np.linspace(0, 1, n_bins + 1)[1:-1]))
    tokens = [f"dt|{i}" for i in range(len(DT_EDGES))]
    for etype, e in edges.items():
        tokens += [f"{etype}|{b}" for b in range(len(e) + 1)]
    return Vocab(tokens, edges)


def encode_client(vocab, etypes, amounts, days):
    ids = [BOS]
    for e, a, d in zip(etypes, amounts, days):
        ids += [vocab.dt(d), vocab.state(e, a)]
    return ids
```

### De onde vem o vocabulário

`vocab.itos` é uma lista de strings; o id de um token é a posição dele nessa lista (`vocab.stoi` é o dicionário inverso). O modelo só vê ids: `nn.Embedding` indexa a linha do id, a string existe pra leitura humana. A lista é montada por `fit_vocab` a partir do treino, em três blocos, nesta ordem:

| bloco | tokens | quantidade | ids |
|-------|--------|-----------:|-----|
| especiais | `<pad>`, `<bos>`, `<unk>` | 3 | 0, 1, 2 |
| estado | `<event_type>\|<faixa>`, um por tipo × faixa | 113 | 3 a 115 |
| intervalo | `dt\|0` a `dt\|14`, um por borda de `DT_EDGES` | 15 | 116 a 130 |

Total: 131. Estado e intervalo saem de um único `sorted(set(tokens))`; como a ordenação é de string e os tipos começam com maiúscula, os `dt\|*` caem no fim.

Os 113 tokens de estado vêm de 23 `event_type` distintos no treino. `fit_vocab` tenta 6 faixas por tipo (5 quantis), mas só cria as faixas quando o tipo tem pelo menos 6 valores distintos não nulos; senão o tipo fica com uma faixa só (`|0`):

+ 18 tipos × 6 faixas = 108 (transações e empréstimos)
+ 5 tipos × 1 faixa = 5: `CARD:classic`, `CARD:gold`, `CARD:junior` (valor sempre 0) e dois tipos raros

Tipo ou faixa que não apareceu no treino vira `<unk>` (id 2) em validação e teste; é o `stoi.get(..., UNK)` em `Vocab.state`.

Bordas de valor (quantis do treino) dos tipos que aparecem:

| event_type           | bordas (5 quantis, 6 faixas)               |
|----------------------|--------------------------------------------|
| PRIJEM:VKLAD         | 2000, 7419, 12903, 19052, 24981            |
| PRIJEM:PREVOD Z UCTU | 3161, 3391, 3993, 23624, 38369             |
| PRIJEM:UROK          | 75.9, 102.9, 129.6, 163.9, 213.3           |
| VYDAJ:VYBER          | 1200, 2400, 4080, 7500, 14700              |

Escala natural dos dados: `abs(amount)`, só valores > 0, só treino (`ts < val_start`). As bordas são os percentis em sextos (`np.quantile` com `linspace(0, 1, 7)[1:-1]`).

```python
train_ev.filter(pl.col("amount") != 0).with_columns(pl.col("amount").abs()).to_pandas() \
    .groupby("event_type")["amount"].describe(percentiles=[i / 6 for i in range(1, 6)])
```

| event_type           |  count |    mean |     std |    min |  16.7% |  33.3% |     50% |   66.7% |   83.3% |     max |
|----------------------|-------:|--------:|--------:|-------:|-------:|-------:|--------:|--------:|--------:|--------:|
| PRIJEM:VKLAD         |  90152 | 15010.0 | 12084.1 |  100.0 | 2000.0 | 7419.0 | 12903.0 | 19052.0 | 24981.0 | 49992.0 |
| PRIJEM:PREVOD Z UCTU |  19547 | 17699.5 | 17584.2 | 2900.0 | 3161.0 | 3391.0 |  3993.0 | 23624.0 | 38369.0 | 74812.0 |
| PRIJEM:UROK          | 102618 |   143.7 |    74.7 |    0.1 |   75.9 |  102.9 |   129.6 |   163.9 |   213.3 |   515.7 |
| VYDAJ:VYBER          | 151002 |  7921.3 |  9699.6 |  100.0 | 1200.0 | 2400.0 |  4080.0 |  7500.0 | 14700.0 | 86400.0 |

Considerando o mesmo cliente da seção anterior:

| ts         | dias | token dt | token estado             |
|------------|-----:|----------|--------------------------|
| 1993-02-26 |    0 | `dt\|0`  | `PRIJEM:VKLAD\|0`         |
| 1993-03-12 |   14 | `dt\|7`  | `PRIJEM:PREVOD Z UCTU\|3` |
| 1993-03-28 |   16 | `dt\|7`  | `PRIJEM:VKLAD\|1`         |
| 1993-03-31 |    3 | `dt\|3`  | `PRIJEM:UROK\|0`          |
| 1993-04-12 |   12 | `dt\|6`  | `PRIJEM:PREVOD Z UCTU\|3` |
| 1993-04-27 |   15 | `dt\|7`  | `VYDAJ:VYBER\|4`          |
| 1993-04-30 |    3 | `dt\|3`  | `PRIJEM:UROK\|2`          |
| 1993-05-12 |   12 | `dt\|6`  | `PRIJEM:PREVOD Z UCTU\|3` |

Saída de `encode_client` para esses 8 eventos: `<bos>` na frente e, para cada evento, o par (token dt, token estado). 8 eventos viram 1 + 2 × 8 = 17 tokens:

```python
['<bos>', 'dt|0', 'PRIJEM:VKLAD|0', 'dt|7', 'PRIJEM:PREVOD Z UCTU|3', 'dt|7', 'PRIJEM:VKLAD|1', 'dt|3', 'PRIJEM:UROK|0', 'dt|6', 'PRIJEM:PREVOD Z UCTU|3', 'dt|7', 'VYDAJ:VYBER|4', 'dt|3', 'PRIJEM:UROK|2', 'dt|6', 'PRIJEM:PREVOD Z UCTU|3']
# ids: [1, 116, 54, 128, 45, 128, 55, 124, 48, 127, 45, 128, 114, 124, 50, 127, 45]   (vocab de 131 tokens)
```

## 3. Janelas e split

Split por data, nunca por cliente: treino `ts < val_start`, validação `val_start <= ts < test_start`, teste `ts >= test_start`. No treino toda posição conta e a sequência é fatiada em janelas de `CTX`. Em validação/teste a janela é o sufixo do histórico e só as posições depois do corte contam na loss (`mask`).

```python
class SeqDataset(Dataset):
    def __init__(self, vocab, events, score_from, ctx, train=False):
        self.items = []
        for _, g in events.group_by("client_id", maintain_order=True):
            ts = g["ts"].to_numpy()
            days = np.concatenate([[0], np.diff(ts).astype("timedelta64[D]").astype(int)])
            ids = np.array(encode_client(vocab, g["event_type"].to_list(), g["amount"].to_list(), days))
            if score_from is None:
                mask = np.ones(len(ids), bool)
            else:
                scored = ts >= np.datetime64(score_from)
                if not scored.any():
                    continue
                mask = np.concatenate([[False], np.repeat(scored, 2)])
            mask[0] = False
            if train:
                for s in range(0, len(ids), ctx):
                    if len(ids) - s > 1:
                        self.items.append((ids[s : s + ctx + 1], mask[s : s + ctx + 1]))
            else:
                self.items.append((ids[-(ctx + 1) :], mask[-(ctx + 1) :]))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


def collate(batch):
    L = max(len(ids) for ids, _ in batch)
    x = torch.zeros(len(batch), L - 1, dtype=torch.long)
    y, m = torch.zeros_like(x), torch.zeros_like(x, dtype=torch.bool)
    for i, (ids, mask) in enumerate(batch):
        n = len(ids) - 1
        x[i, -n:] = torch.as_tensor(ids[:-1])   # pad à esquerda
        y[i, -n:] = torch.as_tensor(ids[1:])    # alvo = próximo token
        m[i, -n:] = torch.as_tensor(mask[1:])
    return x, y, m


train_ev = ev.filter(pl.col("ts") < val_start)
vocab = fit_vocab(train_ev)
train_dl = DataLoader(SeqDataset(vocab, train_ev, None, CTX, train=True), BS, shuffle=True, collate_fn=collate)
val_dl = DataLoader(SeqDataset(vocab, ev.filter(pl.col("ts") < test_start), val_start, CTX), BS, collate_fn=collate)
test_dl = DataLoader(SeqDataset(vocab, ev, test_start, CTX), BS, collate_fn=collate)
```

## 4. Modelo

Normalização de camada (`LayerNorm`) na entrada de cada sub-bloco, não depois da soma com o resíduo. Pesos da camada de saída amarrados aos do embedding. Posição por rotação: os vetores de consulta e de chave de cada token são girados por um ângulo proporcional à posição, então o produto entre eles depende só da distância relativa (Rotary Position Embedding, última referência). Embedding de posição aprendido perde pra uma rede recorrente nesse dado. Máscara aditiva: causal (cada posição só atende às anteriores) e chaves de `<pad>` fora da atenção.

```python
def rope_tables(T, dh, device):
    inv = 1.0 / (10000 ** (torch.arange(0, dh, 2, device=device).float() / dh))
    ang = torch.arange(T, device=device).float()[:, None] * inv[None, :]
    ang = torch.cat([ang, ang], -1)
    return ang.cos(), ang.sin()


def apply_rope(t, cos, sin):
    dh = t.size(-1)
    t1, t2 = t[..., : dh // 2], t[..., dh // 2 :]
    return t * cos + torch.cat([-t2, t1], -1) * sin


def causal_pad_mask(x):
    """[B, 1, T, T] aditiva: i só atende j <= i; chaves de <pad> ficam fora (pad atende a si mesma)."""
    B, T = x.shape
    i = torch.arange(T, device=x.device)
    causal = (i[None, :] <= i[:, None])[None]
    keys_ok = (x != PAD)[:, None, :] | torch.eye(T, dtype=torch.bool, device=x.device)[None]
    return torch.zeros(B, T, T, device=x.device).masked_fill(~(causal & keys_ok), float("-inf"))[:, None]


class Block(nn.Module):
    def __init__(self, d, h, dropout):
        super().__init__()
        self.h = h
        self.qkv, self.proj = nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d), nn.Dropout(dropout))
        self.drop = nn.Dropout(dropout)

    def attn(self, x, mask, rope):
        B, T, d = x.shape
        q, k, v = self.qkv(x).view(B, T, 3, self.h, d // self.h).unbind(2)
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))          # [B, h, T, dh]
        q, k = apply_rope(q, *rope), apply_rope(k, *rope)
        z = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=self.drop.p if self.training else 0)
        return self.proj(z.transpose(1, 2).reshape(B, T, d))

    def forward(self, x, mask, rope):
        x = x + self.drop(self.attn(self.ln1(x), mask, rope))
        return x + self.ff(self.ln2(x))


class Transformer(nn.Module):
    def __init__(self, vocab_size, ctx=CTX, d=128, h=4, layers=2, dropout=0.1):
        super().__init__()
        self.ctx, self.d, self.h = ctx, d, h
        self.tok = nn.Embedding(vocab_size, d)
        self.blocks = nn.ModuleList(Block(d, h, dropout) for _ in range(layers))
        self.ln = nn.LayerNorm(d)
        self.out = nn.Linear(d, vocab_size, bias=False)
        self.out.weight = self.tok.weight
        self.drop = nn.Dropout(dropout)
        self.apply(self._init)

    def _init(self, m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.zeros_(m.bias)

    def embed(self, x):
        """[B, T, d]; a última posição é o embedding do cliente."""
        T = x.size(1)
        assert T <= self.ctx
        h = self.drop(self.tok(x))
        mask, rope = causal_pad_mask(x), rope_tables(T, self.d // self.h, x.device)
        for b in self.blocks:
            h = b(h, mask, rope)
        return self.ln(h)

    def forward(self, x):
        return self.out(self.embed(x))
```

## 5. Treino

Entropia cruzada só nas posições marcadas. Métricas só nos alvos que são estado (os tokens de `dt` também são previstos, mas não são o objeto). Otimizador AdamW, learning rate sobe linearmente na primeira época e fica constante depois, norma do gradiente cortada em 1.0. Salva o melhor por log-verossimilhança negativa (`nll`) de validação.

```python
def lm_loss(logits, y, m):
    return F.cross_entropy(logits[m], y[m])


@torch.no_grad()
def evaluate(model, loader, state_ids):
    model.eval()
    nll = top1 = top5 = n = 0
    for x, y, m in loader:
        x, y, m = x.to(device), y.to(device), m.to(device)
        m = m & torch.isin(y, state_ids)
        lg, t = model(x)[m], y[m]
        ranks = (lg > lg.gather(1, t[:, None])).sum(1) + 1
        nll += F.cross_entropy(lg, t, reduction="sum").item()
        top1 += (ranks <= 1).sum().item()
        top5 += (ranks <= 5).sum().item()
        n += t.numel()
    return {"nll": nll / n, "top1": top1 / n, "top5": top5 / n}


torch.manual_seed(42)
model = Transformer(len(vocab)).to(device)
opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
state_ids = torch.tensor(vocab.state_ids, device=device)
best = float("inf")

for epoch in range(1, EPOCHS + 1):
    model.train()
    for i, (x, y, m) in enumerate(train_dl):
        if epoch <= WARMUP:
            for g in opt.param_groups:
                g["lr"] = LR * (epoch - 1 + i / len(train_dl)) / WARMUP
        x, y, m = x.to(device), y.to(device), m.to(device)
        loss = lm_loss(model(x), y, m)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    val = evaluate(model, val_dl, state_ids)
    print(epoch, val)
    if val["nll"] < best:
        best = val["nll"]
        torch.save(model.state_dict(), "best.pt")
```

## 6. Inferência

Próximo evento de um cliente dado o histórico até uma data. Dois passos porque a sequência é `dt s dt s`: primeiro o intervalo mais provável, depois o estado condicionado a ele.

Entrada: `hist`, a fatia de `ev` de um cliente até a data de corte (as mesmas quatro colunas da seção 1). Dentro de `next_event` ela vira a lista de ids da seção 2, cortada aos últimos `CTX - 1` tokens, e entra no modelo como tensor de forma `[1, T]`.

Saída do modelo: logits de forma `[1, T, len(vocab)]`; só a última posição interessa, `[len(vocab)]`. Saída de `next_event`: o token de intervalo mais provável e a lista dos `k` tokens de estado mais prováveis com a probabilidade, softmax só sobre os ids de estado.

```python
# hist                      polars, n linhas × (client_id, ts, event_type, amount)
# ids                       list[int], len(ids) <= CTX - 1        ex.: [1, 116, 54, 128, 45, ...]
# torch.tensor([ids])       [1, T]
# model(...)                [1, T, 131]        logits, um por token do vocabulário, em cada posição
# logits = model(...)[0, -1]   [131]           só a última posição: distribuição do próximo token
# next_event(...)           (str, list[tuple[str, float]])
```

```python
model.load_state_dict(torch.load("best.pt", map_location=device))
model.eval()


def next_event(model, vocab, hist, k=5):
    """hist: eventos de um cliente até o corte, ordenados por ts."""
    ts = hist["ts"].to_numpy()
    days = np.concatenate([[0], np.diff(ts).astype("timedelta64[D]").astype(int)])
    ids = encode_client(vocab, hist["event_type"].to_list(), hist["amount"].to_list(), days)[-(CTX - 1) :]
    dt_ids, st_ids = torch.tensor(vocab.dt_ids), torch.tensor(vocab.state_ids)
    with torch.no_grad():
        logits = model(torch.tensor([ids], device=device))[0, -1].cpu()
        dt = int(dt_ids[logits[dt_ids].argmax()])
        logits = model(torch.tensor([ids + [dt]], device=device))[0, -1].cpu()
    p, j = logits[st_ids].softmax(0).topk(k)
    return vocab.itos[dt], [(vocab.itos[st_ids[i]], float(pi)) for pi, i in zip(p, j)]


hist = ev.filter((pl.col("client_id") == "2") & (pl.col("ts") < date(1998, 1, 1)))
next_event(model, vocab, hist)
# ('dt|4', [('PRIJEM:PREVOD Z UCTU|3', 0.41), ...])   # ilustrativo
```
