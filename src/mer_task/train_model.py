from datasets import load_from_disk, concatenate_datasets
from transformers import BertTokenizerFast, BertForTokenClassification, TrainingArguments, Trainer, DataCollatorForTokenClassification
from transformers import XLMRobertaTokenizerFast, XLMRobertaForTokenClassification, AutoModelForTokenClassification
from transformers import EarlyStoppingCallback
from transformers import AutoTokenizer, AutoModelForMaskedLM
from seqeval.metrics import classification_report, f1_score
from seqeval.scheme import IOB2
from seqeval.metrics.sequence_labeling import get_entities
import json
import numpy as np
import os
from collections import defaultdict
 
precision_por_entidade_por_municipio = {}
recall_por_entidade_por_municipio = {}
macro_f1_por_municipio = {}

# Carregar mappings
with open("data/output_paper_pt/label_mappings.json") as f:
    mappings = json.load(f)
label2id = mappings["label2id"]
id2label = {int(k): v for k, v in mappings["id2label"].items()}

folder_name = ""

# Tokenizer and model 
'''
tokenizer_name = "xlm-roberta-large"
tokenizer = XLMRobertaTokenizerFast.from_pretrained("xlm-roberta-large")

model = XLMRobertaForTokenClassification.from_pretrained(
        tokenizer_name,
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id
    )
'''

tokenizer_name = "neuralmind/bert-large-portuguese-cased"
tokenizer = BertTokenizerFast.from_pretrained("neuralmind/bert-large-portuguese-cased")

model = BertForTokenClassification.from_pretrained(
        tokenizer_name,
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id
    )

municipalities = ["alandroal", "covilha", "campomaior", "guimaraes", "fundao", "porto"]

# This two are for the LOMO scenario

# train_municipalities = ["alandroal", "campomaior", "covilha", "fundao", "guimaraes"]

# test_municipalities = ["porto"]

datasets_train = [
    load_from_disk(f"data/output_paper_pt/municipio_{m}/train")
    for m in municipalities
]

datasets_val = [
    load_from_disk(f"data/output_paper_pt/municipio_{m}/val")
    for m in municipalities
]

datasets_test = [
    load_from_disk(f"data/output_paper_pt/municipio_{m}/test")
    for m in municipalities
]


train_dataset = concatenate_datasets(datasets_train)
val_dataset = concatenate_datasets(datasets_val)
test_dataset = concatenate_datasets(datasets_test)

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    true_labels, pred_labels = [], []

    for label_seq, pred_seq in zip(labels, preds):
        true_seq, pred_seq_clean = [], []
        for l, p in zip(label_seq, pred_seq):
            if l != -100:
                tl = id2label[l]
                pl = id2label[p]

                true_seq.append(tl)
                pred_seq_clean.append(pl)

        true_labels.append(true_seq)
        pred_labels.append(pred_seq_clean)

    f1_entity = f1_score(true_labels, pred_labels, mode="strict", scheme=IOB2)
    report_entity = classification_report(true_labels, pred_labels, mode="strict", scheme=IOB2, output_dict=True)
    print("\n=== ENTITY-LEVEL ===")
    print(classification_report(true_labels, pred_labels, mode="strict", scheme=IOB2))

    # Add these lines to extract global metrics:
    compute_metrics.last_macro_precision = report_entity.get("macro avg", {}).get("precision", 0.0)
    compute_metrics.last_macro_recall = report_entity.get("macro avg", {}).get("recall", 0.0)
    compute_metrics.last_macro_f1 = report_entity.get("macro avg", {}).get("f1-score", 0.0)

    compute_metrics.last_micro_precision = report_entity.get("micro avg", {}).get("precision", 0.0)
    compute_metrics.last_micro_recall = report_entity.get("micro avg", {}).get("recall", 0.0)
    compute_metrics.last_micro_f1 = report_entity.get("micro avg", {}).get("f1-score", 0.0)

    # Extract entity-level metrics
    entity_f1s = {k: v["f1-score"] for k, v in report_entity.items()
                  if k not in ["macro avg", "weighted avg", "micro avg"]}
    entity_precisions = {k: v["precision"] for k, v in report_entity.items()
                         if k not in ["macro avg", "weighted avg", "micro avg"]}
    entity_recalls = {k: v["recall"] for k, v in report_entity.items()
                      if k not in ["macro avg", "weighted avg", "micro avg"]}

    compute_metrics.last_entity_f1s = entity_f1s
    compute_metrics.last_entity_precisions = entity_precisions
    compute_metrics.last_entity_recalls = entity_recalls

    return {"f1_entity": f1_entity}


# Train
args = TrainingArguments(
        output_dir=f"results/{folder_name}",
        eval_strategy="epoch",
        learning_rate=2e-5,            
        per_device_train_batch_size=2,  
        per_device_eval_batch_size=2,  
        gradient_accumulation_steps=4,
        fp16=True,  
        num_train_epochs=15,
        weight_decay=0.01,
        logging_dir=f"results/{folder_name}",
        logging_strategy="steps",
        logging_steps=10,
        save_strategy="epoch", 
        save_total_limit=5,
        load_best_model_at_end=True,
        metric_for_best_model="f1_entity",
        seed=13 # 13 42 123
    )

data_collator = DataCollatorForTokenClassification(tokenizer)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=4)] 
)

trainer.train()

save_path = f"results/{folder_name}"
os.makedirs(save_path, exist_ok=True)

model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)


eval_results = trainer.predict(test_dataset)
print(f"F1 (entity-level): {eval_results.metrics['test_f1_entity']:.4f}")


preds = eval_results.predictions.argmax(-1)
labels = eval_results.label_ids

true_labels = []
pred_labels = []

for label_seq, pred_seq in zip(labels, preds):
    true_seq = []
    pred_seq_clean = []
    for l, p in zip(label_seq, pred_seq):
        if l != -100: 
            true_seq.append(id2label[l])
            pred_seq_clean.append(id2label[p])
    true_labels.append(true_seq)
    pred_labels.append(pred_seq_clean)

f1s = compute_metrics.last_entity_f1s
precisions = compute_metrics.last_entity_precisions
recalls = compute_metrics.last_entity_recalls

precision_por_entidade_por_municipio["GLOBAL"] = precisions
recall_por_entidade_por_municipio["GLOBAL"] = recalls
macro_f1_por_municipio["GLOBAL"] = float(np.mean(list(f1s.values())))

global_metrics = {
    "macro_precision": float(compute_metrics.last_macro_precision),
    "macro_recall": float(compute_metrics.last_macro_recall),
    "macro_f1": float(compute_metrics.last_macro_f1),
    "micro_precision": float(compute_metrics.last_micro_precision),
    "micro_recall": float(compute_metrics.last_micro_recall),
    "micro_f1": float(compute_metrics.last_micro_f1)
}

base = f"results/{folder_name}"
os.makedirs(base, exist_ok=True)


with open(f"{base}/global_metrics.json", "w", encoding="utf-8") as f:
    json.dump(global_metrics, f, ensure_ascii=False, indent=2)

# F1 médio por entidade
f1_medio_por_entidade = {ent: float(np.mean([val])) for ent, val in f1s.items()}

with open(f"{base}/f1_mean_per_entity.json", "w", encoding="utf-8") as f:
    json.dump(f1_medio_por_entidade, f, ensure_ascii=False, indent=2)

with open(f"{base}/precision_per_entity.json", "w", encoding="utf-8") as f:
    json.dump(precision_por_entidade_por_municipio, f, ensure_ascii=False, indent=2)

with open(f"{base}/recall_per_entity.json", "w", encoding="utf-8") as f:
    json.dump(recall_por_entidade_por_municipio, f, ensure_ascii=False, indent=2)

with open(f"{base}/macro_f1_per_municipio.json", "w", encoding="utf-8") as f:
    json.dump(macro_f1_por_municipio, f, ensure_ascii=False, indent=2)

predictions_path = f"{base}/predictions.jsonl"
with open(predictions_path, "w", encoding="utf-8") as f:
    for label_seq, pred_seq, input_ids in zip(labels, preds, test_dataset["input_ids"]):
        tokens = tokenizer.convert_ids_to_tokens(input_ids)

        tokens_clean = []
        true_seq = []
        pred_seq_clean = []

        for tok, l, p in zip(tokens, label_seq, pred_seq):
            if l != -100: 
                tokens_clean.append(tok)
                true_seq.append(id2label[l])
                pred_seq_clean.append(id2label[p])

        f.write(json.dumps({
            "tokens": tokens_clean,
            "true": true_seq,
            "pred": pred_seq_clean
        }, ensure_ascii=False) + "\n")


