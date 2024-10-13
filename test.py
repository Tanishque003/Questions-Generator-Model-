import pandas as pd
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# Load the fine-tuned model and tokenizer
model = T5ForConditionalGeneration.from_pretrained("./fine_tuned_t5")
tokenizer = T5Tokenizer.from_pretrained("./fine_tuned_t5")

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Load the test dataset (Assumes it has 'job_title', 'question', and 'answer' columns)
test_df = pd.read_csv('test.csv')

# Function to generate questions from the job titles
def generate_question(job_title, model, tokenizer):
    input_text = f"Generate a question for the job title: {job_title}"
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(device)
    
    # Generate output with beam search
    outputs = model.generate(input_ids, max_length=50, num_beams=5, early_stopping=True)
    
    # Decode the generated question
    generated_question = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_question

# Function to calculate exact match accuracy
def exact_match_accuracy(generated, expected):
    return int(generated.strip().lower() == expected.strip().lower())

# Function to calculate BLEU score for a generated question
def calculate_bleu(generated, expected):
    reference = expected.split()  # Split expected question into tokens
    candidate = generated.split()  # Split generated question into tokens
    smoothing = SmoothingFunction().method4  # Smoothing to handle short sentences
    return sentence_bleu([reference], candidate, smoothing_function=smoothing)

# Test the model on the dataset and calculate accuracy metrics
def evaluate_model(test_df, model, tokenizer):
    total_samples = len(test_df)
    exact_match_count = 0
    bleu_scores = []
    
    for idx, row in test_df.iterrows():
        job_title = row['job_title']
        expected_question = row['question']
        
        # Generate a question for the job title
        generated_question = generate_question(job_title, model, tokenizer)
        
        # Calculate exact match accuracy
        exact_match_count += exact_match_accuracy(generated_question, expected_question)
        
        # Calculate BLEU score
        bleu_score = calculate_bleu(generated_question, expected_question)
        bleu_scores.append(bleu_score)
        
        # Optionally, print for debugging or to check individual samples
        print(f"Job Title: {job_title}")
        print(f"Generated Question: {generated_question}")
        print(f"Expected Question: {expected_question}")
        print(f"BLEU Score: {bleu_score:.2f}\n")

    # Calculate average BLEU score
    avg_bleu_score = sum(bleu_scores) / total_samples
    exact_match_accuracy_percent = (exact_match_count / total_samples) * 100

    print(f"Exact Match Accuracy: {exact_match_accuracy_percent:.2f}%")
    print(f"Average BLEU Score: {avg_bleu_score:.4f}")

# Run the evaluation
evaluate_model(test_df, model, tokenizer)
