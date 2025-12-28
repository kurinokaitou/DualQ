import argparse
from dataclasses import dataclass
from typing import List

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

from model_utils import inject_peft_model


@dataclass
class CommonsenseExample:
    prompt: str


def build_prompt(question: str, choices: List[str]) -> str:
    labels = ["A", "B", "C", "D", "E"]
    options = "\n".join(f"{label}. {choice}" for label, choice in zip(labels, choices))
    return f"Question: {question}\nChoices:\n{options}\nAnswer:"


def preprocess(example: dict) -> CommonsenseExample:
    question = example["question"]
    choices = example["choices"]["text"]
    answer_key = example["answerKey"]
    labels = ["A", "B", "C", "D", "E"]
    answer_index = labels.index(answer_key)
    prompt = build_prompt(question, choices)
    completion = f" {labels[answer_index]}"
    return CommonsenseExample(prompt=prompt + completion)


def tokenize_batch(batch: dict, tokenizer: AutoTokenizer, max_length: int) -> dict:
    texts = batch["prompt"]
    tokens = tokenizer(
        texts,
        max_length=max_length,
        padding="max_length",
        truncation=True,
    )
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="meta-llama/Llama-2-7b-hf")
    parser.add_argument("--output_dir", default="./outputs")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--group_size", type=int, default=128)
    parser.add_argument("--bit", type=int, default=4)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16)
    parser.add_argument("--num_train_epochs", type=float, default=1)
    args = parser.parse_args()

    dataset = load_dataset("commonsense_qa")
    processed = dataset["train"].map(preprocess, remove_columns=dataset["train"].column_names)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenized = processed.map(
        tokenize_batch,
        batched=True,
        fn_kwargs={"tokenizer": tokenizer, "max_length": args.max_length},
        remove_columns=processed.column_names,
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )

    model = inject_peft_model(model, group_size=args.group_size, bit=args.bit)

    for name, param in model.named_parameters():
        param.requires_grad = "gamma_" in name

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        logging_steps=10,
        save_steps=500,
        fp16=True,
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    trainer.train()


if __name__ == "__main__":
    main()
