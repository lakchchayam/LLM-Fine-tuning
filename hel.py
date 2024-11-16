# Import necessary libraries
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, TextDataset, DataCollatorForLanguageModeling

# Step 1: Data Preprocessing Pipeline
def preprocess_data(input_file, output_dir):
    # Read and preprocess the text data to optimize quality
    with open(input_file, 'r') as f:
        text_data = f.readlines()
    
    # Remove unwanted characters and improve text quality (simplification can be added)
    processed_text = [line.strip().lower() for line in text_data if line.strip()]
    
    # Save preprocessed data
    with open(output_dir + '/preprocessed_data.txt', 'w') as f:
        for line in processed_text:
            f.write(line + '\n')

# Step 2: Load Pre-trained Model and Tokenizer
model_name = "gpt2"  # Example model, you can choose another pre-trained model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Step 3: Prepare the Dataset for Training
def load_dataset(file_path, tokenizer, block_size=128):
    # Load and tokenize the dataset
    dataset = TextDataset(
        tokenizer=tokenizer,
        file_path=file_path,
        block_size=block_size
    )
    return dataset

# Step 4: Set Up Data Collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False  # Since it's a causal language model, no masked language modeling
)

# Step 5: Fine-tuning Setup with Trainer
def fine_tune_model(model, tokenizer, train_dataset):
    training_args = TrainingArguments(
        output_dir="./results",           # Output directory for model checkpoints
        overwrite_output_dir=True,       # Overwrite the output directory
        num_train_epochs=3,              # Set number of training epochs
        per_device_train_batch_size=4,   # Batch size per device (GPU/CPU)
        save_steps=500,                  # Save checkpoint every 500 steps
        save_total_limit=2,              # Limit the number of saved checkpoints
        logging_steps=100,               # Log training progress every 100 steps
    )
    
    trainer = Trainer(
        model=model, 
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset
    )
    
    trainer.train()

# Step 6: Evaluation and Testing
def evaluate_model(model, tokenizer, test_data):
    model.eval()
    inputs = tokenizer(test_data, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model.generate(inputs['input_ids'])
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Main Function
if __name__ == "__main__":
    # File paths
    input_file = 'data/raw_text_data.txt'
    output_dir = 'data'
    
    # Preprocess the raw text data
    preprocess_data(input_file, output_dir)
    
    # Load the processed data
    train_dataset = load_dataset(output_dir + '/preprocessed_data.txt', tokenizer)
    
    # Fine-tune the model
    fine_tune_model(model, tokenizer, train_dataset)
    
    # Test the fine-tuned model with some example input
    test_input = "How do you train a language model?"
    output = evaluate_model(model, tokenizer, test_input)
    
    print(f"Model Output: {output}")
