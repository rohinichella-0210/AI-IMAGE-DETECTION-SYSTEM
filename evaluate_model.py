from datasets import load_dataset
from transformers import AutoImageProcessor, AutoModelForImageClassification, Trainer, TrainingArguments
import torch
import numpy as np
import evaluate

# 1. Configuration
MODEL_PATH = "./truthlens-custom-model"
# Choose which images to test on (e.g., the first 500)
START_INDEX = 0
END_INDEX = 500

print(f"Loading Test Dataset ({START_INDEX} to {END_INDEX})...")
dataset = load_dataset("imagefolder", data_dir="dataset/train", split="train")
# Shuffle and select a test slice
test_ds = dataset.shuffle(seed=42).select(range(START_INDEX, END_INDEX))

print("Loading Model and Processor...")
processor = AutoImageProcessor.from_pretrained("umm-maybe/AI-image-detector")
model = AutoModelForImageClassification.from_pretrained(MODEL_PATH)

# 2. Image Formatting
def transforms(examples):
    inputs = processor([img.convert("RGB") for img in examples["image"]], return_tensors="pt")
    inputs["labels"] = examples["label"]
    return inputs

test_ds.set_transform(transforms)

def collate_fn(batch):
    return {
        'pixel_values': torch.stack([x['pixel_values'] for x in batch]),
        'labels': torch.tensor([x['labels'] for x in batch])
    }

# 3. Metric Setup
metric = evaluate.load("accuracy")
def compute_metrics(p):
    return metric.compute(predictions=np.argmax(p.predictions, axis=1), references=p.label_ids)

# 4. Run Evaluation
trainer = Trainer(
    model=model,
    args=TrainingArguments(output_dir="./temp", remove_unused_columns=False, per_device_eval_batch_size=4),
    data_collator=collate_fn,
    compute_metrics=compute_metrics,
)

print("\n" + "="*30)
results = trainer.evaluate(test_ds)
print(f"MODEL ACCURACY: {results['eval_accuracy'] * 100:.2f}%")
print(f"EVALUATION LOSS: {results['eval_loss']:.4f}")
print("="*30)