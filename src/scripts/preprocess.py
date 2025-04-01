
from datasets import load_dataset
from transformers import PreTrainedTokenizer
from loguru import logger

def load_and_tokenize_data(tokenizer, train_split="train[:80]", test_split="test[:10]"):
    logger.info("-------load_and_tokenize_data------------------")

    train_dataset = load_dataset("amazon_polarity", split=train_split)
    test_dataset = load_dataset("amazon_polarity", split=test_split)
    print(train_dataset[0])
    print(f"Train dataset length: {len(train_dataset)}")
    print(f"Test dataset length: {len(test_dataset)}")
    
    def tokenize_function(example):
        return tokenizer(example["content"], padding="max_length", truncation=True, max_length=256)
    
    tokenized_train = train_dataset.map(tokenize_function, batched=True)
    tokenized_test = test_dataset.map(tokenize_function, batched=True)
    
    columns_to_keep = ["input_ids", "attention_mask", "label"]
    tokenized_train = tokenized_train.remove_columns([col for col in tokenized_train.column_names if col not in columns_to_keep])
    tokenized_test = tokenized_test.remove_columns([col for col in tokenized_test.column_names if col not in columns_to_keep])
    
    print("Tokenized train columns:", tokenized_train.column_names)
    print("Tokenized test columns:", tokenized_test.column_names)
    return tokenized_train, tokenized_test