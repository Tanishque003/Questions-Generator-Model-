import pandas as pd
from datasets import Dataset
from transformers import T5Tokenizer, T5ForConditionalGeneration, Seq2SeqTrainingArguments, Seq2SeqTrainer
import torch

# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load the CSV files into pandas DataFrames
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')
val_df = pd.read_csv('validation.csv')

# Convert DataFrames into Hugging Face Dataset format
def load_dataset_from_df(df):
    dataset = Dataset.from_pandas(df)
    return dataset

train_dataset = load_dataset_from_df(train_df)
test_dataset = load_dataset_from_df(test_df)
val_dataset = load_dataset_from_df(val_df)

# Initialize the tokenizer and model
model_name = "t5-small"  # You can also use 't5-base' or 't5-large'
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name).to(device)  # Move model to GPU

# Preprocessing function for the dataset
def preprocess_data(examples):
    # Input: Job title and Answer
    inputs = ["Generate a question for the job title: " + job for job in examples['job_title']]
    targets = examples['question']  # The model's target output is the question
    
    # Tokenizing inputs and targets
    model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding="max_length")
    
    # Tokenize target labels (questions)
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=128, truncation=True, padding="max_length")
    
    # Move inputs and labels to the GPU
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# Apply the preprocessing to the datasets
train_dataset = train_dataset.map(preprocess_data, batched=True)
test_dataset = test_dataset.map(preprocess_data, batched=True)
val_dataset = val_dataset.map(preprocess_data, batched=True)

# Set up training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir="./results",  # Directory where model checkpoints will be saved
    evaluation_strategy="epoch",  # Evaluate at the end of every epoch
    learning_rate=5e-5,  # Learning rate
    per_device_train_batch_size=4,  # Adjust based on your system memory
    per_device_eval_batch_size=4,
    weight_decay=0.01,
    save_total_limit=3,  # Limit the number of saved checkpoints
    num_train_epochs=3,  # Set the number of training epochs
    predict_with_generate=True,  # Use model generate for predictions during eval
    logging_dir="./logs",  # Directory for logs
    logging_steps=100,  # Log every 100 steps
)

# Initialize the Trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
)

# Train the model
trainer.train()

# Save the fine-tuned model and tokenizer
model.save_pretrained("./fine_tuned_t5")
tokenizer.save_pretrained("./fine_tuned_t5")
