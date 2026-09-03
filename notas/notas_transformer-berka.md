# Transformer sobre transações bancárias (Berka)

Cada cliente vira uma sequência de tokens (compra, salário, empréstimo...) e um Transformer causal aprende a prever o próximo estado. Código em `knowledge-base/lab-transformers/` (repo privado); esta nota é o roteiro mínimo pra reproduzir.

## 1. Setup

```bash
cd lab-transformers
uv sync                     # Python >= 3.12, torch, polars, kagglehub
```

Credencial do Kaggle em `~/.kaggle/kaggle.json` (ou `KAGGLE_API_TOKEN` no ambiente).

## 2. Dados

```bash
uv run python scripts/get_data.py berka      # kagglehub marceloventura/the-berka-dataset -> data/raw/berka/
```

Na primeira vez que o treino toca o dataset, `lt/data.py` monta a tabela canônica `(client_id, ts, event_type, amount)` a partir de `trans.csv`, `loan.csv`, `card.csv` + `disp.csv` e grava `data/processed/berka.parquet`.

Tokenização (`lt/tokenizer.py`): cada evento vira o par `dt|<bucket>` (dias desde o evento anterior, buckets fixos) + `<tipo>|<faixa>` (faixa de valor por quantil, ajustada só no treino). Sequência: `<bos> dt s dt s ...`. Split por corte temporal global (`val_start`, `test_start`), nunca por cliente.

## 3. Treino

Config `configs/tfm_berka_max.yaml`:

```yaml
name: tfm_berka_max
seed: 42
dataset: berka
val_start: 1997-07-01
test_start: 1998-01-01
amount_bins: 6
ctx: 512
batch_size: 64
model: transformer
model_args: {ctx: 512, d: 128, h: 4, layers: 2, dropout: 0.1, pos: rope, pad_mask: true}
lr: 1.0e-3
wd: 0.01
clip: 1.0
amp: true
epochs: 30
warmup_epochs: 1
```

```bash
uv run python train.py configs/tfm_berka_max.yaml    # -> runs/tfm_berka_max-<ts>/{config.yaml,vocab.json,metrics.jsonl,best.pt,last.pt}
uv run python eval.py runs/tfm_berka_max-<ts>          # -> eval.json (NLL, top-1, top-5, MRR no próximo estado)
```

Decoder-only GPT pequeno, RoPE como posição (posição aprendida absoluta perdia pro GRU), máscara de padding. Na GPU do RunPod, ~3 min:

```bash
uv run python scripts/remote_train.py train configs/tfm_berka_max.yaml --max-hours 1   # RUNPOD_API_KEY no .env; já roda o eval
```

## 4. Inferência

```bash
uv run python scripts/predict.py runs/tfm_berka_max-<ts> 2 --until 1998-01-01
```

Imprime os últimos 5 eventos da conta, o top-3 do próximo intervalo (`dt`), o top-5 do próximo estado e, se houver, o evento real depois de `--until`.
