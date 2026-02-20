from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
import torch

# Loading small subset of Sci/Tech news
dataset = load_dataset("ag_news", split="train[:500]") # only 500 examples for speed
tech_dataset = [item for item in dataset if item["label"] == 3] # label 3 = Sci/Tech

texts = [item["text"] for item in tech_dataset]

model_name = "tiiuae/falcon-7b-instruct"  # Instruct model we are going to use

# Loading tokenizer from the pretrained model
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Specifying padding token
tokenizer.pad_token = tokenizer.eos_token

# We are downloanding pre-trained weights and instantiating the correct model architecture
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    # MPS (Apple GPU backend) often fails with auto 
    # so we don't allow Hugging Face the model with automatic device placement, so the model stays on CPU
    device_map=None,
    # We set lower precision (not float32 as by default) to be able to run on local machine
    dtype=torch.float16
)

# We are untiying Input embedding matrix and Output projection matrix
model.config.tie_word_embeddings = False 

# The Hugging Face Trainer expects the data to behave like a PyTorch Dataset 
# so we are defining the wrapper class
# The dataset returns as tensors: 
#   - input_ids (integer indices into the model’s embedding matrix), 
#   - attention_mask, and 
#   - labels 
class SimpleDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings
        # set labels equal to input_ids for causal LM
        self.encodings["labels"] = self.encodings["input_ids"]

    def __len__(self):
        return len(self.encodings["input_ids"])

    def __getitem__(self, idx):
        # index each list in the batch
        return {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
    
tokenized = tokenizer(
    texts,
    truncation=True,
    padding="max_length",
    max_length=64
)

# Wrapping as Dataset
dataset_tokenized = SimpleDataset(tokenized)

# LoRA configuration (fast fine-tuning)
lora_config = LoraConfig(
    r=8, # intristic rank
    lora_alpha=16, # scaling factor
    target_modules = ["query_key_value"], # inserting LoRA into query key value layer
    lora_dropout=0.05,
    bias="none", # we don't train bias
    task_type="CAUSAL_LM" # specifying to peft that we are doing next-token prediction
)

# Parameter efficient fine-tuning
model = get_peft_model(model, lora_config)

# Training setup
training_args = TrainingArguments(
    output_dir="./ft_model",
    per_device_train_batch_size=1, # how many examples are processed on each GPU/CPU at a time
    num_train_epochs=1, # number of times the trainer iterates over the entire dataset
    learning_rate=3e-4,
    logging_steps=10, # log info is stored each 10 steps
    save_steps=50 # The model checkpoint on every 50 steps
)

# Defining trainer
trainer = Trainer(
    model=model,
    train_dataset=dataset_tokenized,
    args=training_args
)

# training the model
trainer.train()