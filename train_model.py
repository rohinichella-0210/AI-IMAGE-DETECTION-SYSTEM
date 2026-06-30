import os
import zipfile
import torch
import numpy as np
import evaluate
from datasets import load_dataset
from transformers import AutoImageProcessor, AutoModelForImageClassification, TrainingArguments, Trainer
from PIL import Image

# --- 1. UNZIP THE DATASET ---
# This matches the path in your Downloads folder
zip_path = r"C:\Users\rohin\Downloads\data.zip"
extract_to = "./data_unzipped"

# Check if we need to unzip
if not os.path.exists(extract_to):
    if os.path.exists(zip_path):
        print(f"📦 Found {zip_path}. Unzipping... this may take a minute.")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        print("✅ Unzipped successfully!")
    else:
        print(f"❌ ERROR: Could not find '{zip_path}'")
        print("Please make sure 'data.zip' is in your Downloads folder.")
        exit()

# --- 2. FIND THE CORRECT FOLDER PATH ---
base_path = extract_to
for root, dirs, files in os.walk(extract_to):
    if "train" in dirs and "test" in dirs:
        base_path = root
        break

print(f"✅ Training images found at: {base_path}")

# --- 3. LOAD THE IMAGES ---
try:
    dataset = load_dataset("imagefolder", data_dir=base_path)
    
    train_ds = dataset['train']
    val_key = 'validation' if 'validation' in dataset else 'test'
    if 'valid' in dataset: val_key = 'valid'
    
    val_ds = dataset[val_key]
    test_ds = dataset['test']
except Exception as e:
    print(f"❌ FAILED TO LOAD IMAGES: {e}")
    exit()

print(f"📊 Dataset: Train({len(train_ds)}) | Val({len(val_ds)}) | Test({len(test_ds)})")

# --- 4. SETUP THE MODEL (Vision Transformer) ---
model_name = "umm-maybe/AI-image-detector"
processor = AutoImageProcessor.from_pretrained(model_name)

id2label = {0: "FAKE", 1: "REAL"}
label2id = {"FAKE": 0, "REAL": 1}

model = AutoModelForImageClassification.from_pretrained(
    model_name,
    label2id=label2id,
    id2label=id2label,
    ignore_mismatched_sizes=True
)

# --- 5. IMAGE PREPARATION ---
def transform_images(examples):
    inputs = processor([img.convert("RGB") for img in examples["image"]], return_tensors="pt")
    inputs["labels"] = examples["label"]
    return inputs

train_ds.set_transform(transform_images)
val_ds.set_transform(transform_images)
test_ds.set_transform(transform_images)

def collate_fn(batch):
    return {
        'pixel_values': torch.stack([x['pixel_values'] for x in batch]),
        'labels': torch.tensor([x['labels'] for x in batch])
    }

metric = evaluate.load("accuracy")
def compute_metrics(p):
    return metric.compute(predictions=np.argmax(p.predictions, axis=1), references=p.label_ids)

# --- 6. TRAINING ARGUMENTS ---
training_args = TrainingArguments(
    output_dir="./midjourney-detector",
    remove_unused_columns=False,
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=4, 
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    report_to="none",
)

# --- 7. START TRAINING ---
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=collate_fn,
    train_dataset=train_ds,
    eval_dataset=val_ds, 
    compute_metrics=compute_metrics,
)

print("🚀 Starting Fine-Tuning on Midjourney data...")
trainer.train()

# --- 8. FINAL EVALUATION ---
print("\nRunning final test on the 'test' folder...")
test_results = trainer.evaluate(test_ds)
accuracy = test_results.get('eval_accuracy', 0)
print(f"🎯 FINAL TEST ACCURACY: {accuracy:.2%}")

# --- 9. SAVE THE MODEL ---
trainer.save_model("./midjourney-detector")
processor.save_pretrained("./midjourney-detector")
print("\n✨ SUCCESS! Your model is saved in the 'midjourney-detector' folder.")
# --- END OF FILE ---