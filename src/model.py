from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import concatenate_datasets
import os
import torch

#used datasets
dataset_1 = load_dataset("marmikpandya/mental-health")
dataset_2 = load_dataset("fadodr/mental_health_therapy")
dataset_3 = load_dataset("Amod/mental_health_counseling_conversations")
dataset_4 = load_dataset("jkhedri/psychology-dataset")
dataset_5 = load_dataset("samhog/psychology-6k")
dataset_6 = load_dataset("RAJJ18/mental_health_dataset")
dataset_6_selected = dataset_6["train"].shuffle(seed=42).select(range(3000))

#making the column name uniform
dataset_1 = dataset_1.rename_columns({
    "input": "input",
    "output": "output"
})

dataset_2 = dataset_2.rename_columns({
    "input": "input",
    "output": "output"
})

dataset_3 = dataset_3.rename_columns({
    "Context": "input",
    "Response": "output"
})

dataset_4 = dataset_4.rename_columns({
    "question": "input",
    "response_j": "output",
})

dataset_5 = dataset_5.rename_columns({
    "input": "input",
    "output": "output"
})

dataset_6_selected = dataset_6_selected.rename_columns({
    "input": "input",
    "output": "output"
})

#select only the columns needed (input and output)
dataset_1 = dataset_1.select_columns(["input", "output"])
dataset_2 = dataset_2.select_columns(["input", "output"])
dataset_3 = dataset_3.select_columns(["input", "output"])
dataset_4 = dataset_4.select_columns(["input", "output"])
dataset_5 = dataset_5.select_columns(["input", "output"])
dataset_6_selected = dataset_6_selected.select_columns(["input", "output"])


#load tokenizer
tokenizer = AutoTokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token

def tokenize_function(examples):
    inputs = tokenizer(examples['input'], truncation=True, padding='max_length', max_length=256)
    outputs = tokenizer(examples['output'], truncation=True, padding='max_length', max_length=256)
    
    inputs['labels'] = outputs['input_ids'] 
    return inputs

# Apply the updated tokenize function to your datasets
tokenized_dataset_1 = dataset_1.map(tokenize_function, batched=True)
tokenized_dataset_2 = dataset_2.map(tokenize_function, batched=True)
tokenized_dataset_3 = dataset_3.map(tokenize_function, batched=True)
tokenized_dataset_4 = dataset_4.map(tokenize_function, batched=True)
tokenized_dataset_5 = dataset_5.map(tokenize_function, batched=True)
tokenized_dataset_6 = dataset_6_selected.map(tokenize_function, batched=True)

# Concatenate the datasets
combined_tokenized_dataset = concatenate_datasets([tokenized_dataset_1["train"],
                                                   tokenized_dataset_2["train"],
                                                   tokenized_dataset_3["train"],
                                                   tokenized_dataset_4["train"],
                                                   tokenized_dataset_5["train"],
                                                   tokenized_dataset_6])

# Verify the length of the combined dataset
print(len(combined_tokenized_dataset))


#finetuning the model
# Check if CUDA is available
cuda_available = torch.cuda.is_available()
print(f"CUDA Available: {cuda_available}")

device = torch.device("cuda" if cuda_available else "cpu")
print(f"Using device: {device}")

# If CUDA is available, print the GPU name
if cuda_available:
    current_device = torch.cuda.current_device()
    device_name = torch.cuda.get_device_name(current_device)
    print(f"Current CUDA device: {device_name}")
else:
    print("Using CPU instead.")

model = AutoModelForCausalLM.from_pretrained('gpt2')
model.to(device)

#training argument define
training_args = TrainingArguments(
    output_dir = '/home/pranil/python_projects/gpt2_finetuned/results',
    eval_strategy = 'steps',  # Evaluate every few steps instead of just after each epoch
    save_steps = 500,         # Save checkpoints every 500 steps
    eval_steps = 500,         # Evaluate every 500 steps
    num_train_epochs = 7,
    per_device_train_batch_size = 1,
    per_device_eval_batch_size = 1,
    warmup_steps = 100,
    weight_decay = 0.01,
    logging_dir = './logs',
    report_to = 'none',
    learning_rate = 3e-5,     # Use a lower learning rate for fine-tuning
    logging_steps = 50,       # Log training progress every 50 steps
    fp16=True  # Use mixed precision training
)

#initialize trainer
trainer = Trainer(
    model = model,
    args = training_args,
    train_dataset = combined_tokenized_dataset,
    eval_dataset = tokenized_dataset_2["test"]
    # eval_dataset = combined_tokenized_dataset['train'].train_test_split(test_size = 0.2)
)

#train the model
trainer.train(resume_from_checkpoint='/home/pranil/python_projects/gpt2_finetuned/results/checkpoint-17500')

model_output_dir = '/home/pranil/python_projects/gpt2_finetuned/results/model'
#create the directory if it doesnot exist
os.makedirs(model_output_dir, exist_ok = True)

model.save_pretrained(model_output_dir)
tokenizer.save_pretrained(model_output_dir)
