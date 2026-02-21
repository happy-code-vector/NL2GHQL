"""
Unsloth Fine-Tuning Script for Hermes Miner

Fine-tunes Qwen2.5-Coder-7B for NL2GraphQL task using LoRA.
"""

import os
import sys
import json
import torch
from dataclasses import dataclass
from typing import Optional, List, Dict
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config.settings import config, ModelConfig


@dataclass
class TrainingConfig:
    """Training configuration"""
    # Model settings
    model_name: str = "unsloth/Qwen2.5-Coder-7B-Instruct"
    max_seq_length: int = 4096

    # LoRA settings
    lora_r: int = 16
    lora_alpha: int = 16
    lora_dropout: float = 0.0
    target_modules: List[str] = None

    # Training settings
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    warmup_steps: int = 10
    max_steps: int = 500
    learning_rate: float = 2e-4
    logging_steps: int = 1
    save_steps: int = 100

    # Data settings
    train_dataset: str = "data/datasets/train_dataset.jsonl"
    output_dir: str = "models/hermes_miner"

    def __post_init__(self):
        if self.target_modules is None:
            self.target_modules = [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ]


# Prompt template for training
ALPACA_PROMPT = """### Instruction:
Convert this natural language question into a valid GraphQL query based on the provided schema context.

### Schema Context:
{}

### Question:
{}

### GraphQL Query:
{}"""

EOS_TOKEN = "<|im_end|>"


def load_model(config: TrainingConfig):
    """Load the base model with Unsloth"""
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config.model_name,
        max_seq_length=config.max_seq_length,
        dtype=None,  # Auto-detect
        load_in_4bit=True,  # Use 4-bit quantization for memory efficiency
    )

    return model, tokenizer


def add_lora_adapters(model, config: TrainingConfig):
    """Add LoRA adapters to the model"""
    model = model.get_peft_model(
        model,
        r=config.lora_r,
        target_modules=config.target_modules,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
        use_rslora=False,
        loftq_config=None,
    )
    return model


def format_prompts(examples, tokenizer):
    """Format training examples into prompts"""
    instructions = examples["instruction"]
    inputs = examples["input"]
    outputs = examples["output"]

    texts = []
    for instruction, input_text, output in zip(instructions, inputs, outputs):
        # Must add EOS_TOKEN
        text = ALPACA_PROMPT.format(input_text, instruction, output) + EOS_TOKEN
        texts.append(text)

    return {"text": texts}


def load_dataset(dataset_path: str):
    """Load training dataset"""
    from datasets import load_dataset as hf_load_dataset

    dataset = hf_load_dataset("json", data_files=dataset_path, split="train")
    return dataset


def train(config: TrainingConfig):
    """Main training function"""
    from unsloth import FastLanguageModel
    from trl import SFTTrainer
    from transformers import TrainingArguments
    from datasets import load_dataset as hf_load_dataset

    print("=" * 60)
    print("Hermes Miner Fine-Tuning")
    print("=" * 60)
    print(f"Model: {config.model_name}")
    print(f"Dataset: {config.train_dataset}")
    print(f"Output: {config.output_dir}")
    print("=" * 60)

    # 1. Load Model
    print("\n[1/5] Loading model...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config.model_name,
        max_seq_length=config.max_seq_length,
        dtype=None,
        load_in_4bit=True,
    )
    print(f"Model loaded. Vocab size: {len(tokenizer)}")

    # 2. Add LoRA Adapters
    print("\n[2/5] Adding LoRA adapters...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=config.lora_r,
        target_modules=config.target_modules,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
        use_rslora=False,
        loftq_config=None,
    )
    print("LoRA adapters added.")

    # 3. Load Dataset
    print("\n[3/5] Loading dataset...")
    dataset = hf_load_dataset("json", data_files=config.train_dataset, split="train")
    print(f"Dataset loaded: {len(dataset)} examples")

    # Format prompts
    dataset = dataset.map(
        lambda examples: format_prompts(examples, tokenizer),
        batched=True,
    )
    print("Prompts formatted.")

    # 4. Setup Trainer
    print("\n[4/5] Setting up trainer...")
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=config.max_seq_length,
        dataset_num_proc=2,
        packing=False,  # Short sequences
        args=TrainingArguments(
            per_device_train_batch_size=config.per_device_train_batch_size,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            warmup_steps=config.warmup_steps,
            max_steps=config.max_steps,
            learning_rate=config.learning_rate,
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            logging_steps=config.logging_steps,
            save_steps=config.save_steps,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=3407,
            output_dir=config.output_dir,
            report_to="none",  # Disable wandb/tensorboard
        ),
    )
    print("Trainer configured.")

    # 5. Train
    print("\n[5/5] Starting training...")
    trainer_stats = trainer.train()
    print("Training completed!")
    print(f"Training time: {trainer_stats.metrics['train_runtime']:.2f} seconds")

    # 6. Save Model
    print("\n[6/6] Saving model...")
    output_path = Path(config.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save merged model for vLLM
    model.save_pretrained_merged(
        str(output_path),
        tokenizer,
        save_method="merged_16bit",
    )
    print(f"Model saved to: {output_path}")

    # Also save LoRA adapters separately
    lora_path = output_path / "lora_adapters"
    model.save_pretrained(str(lora_path))
    tokenizer.save_pretrained(str(lora_path))
    print(f"LoRA adapters saved to: {lora_path}")

    return trainer_stats


def test_inference(model_path: str, test_prompts: List[str]):
    """Test inference with the fine-tuned model"""
    from unsloth import FastLanguageModel

    print("\n" + "=" * 60)
    print("Testing Inference")
    print("=" * 60)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=False,
    )
    FastLanguageModel.for_inference(model)

    for prompt in test_prompts:
        print(f"\nPrompt: {prompt}")
        print("-" * 40)

        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.1,
            top_p=0.95,
            do_sample=True,
        )
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(response)


def main():
    """CLI entry point"""
    import argparse

    parser = argparse.ArgumentParser(description="Fine-tune Hermes miner model")
    parser.add_argument("--config", type=str, help="Path to config JSON")
    parser.add_argument("--dataset", type=str, help="Path to training dataset")
    parser.add_argument("--output", type=str, help="Output directory")
    parser.add_argument("--model", type=str, help="Base model name")
    parser.add_argument("--max-steps", type=int, help="Maximum training steps")
    parser.add_argument("--test", action="store_true", help="Run test inference")

    args = parser.parse_args()

    # Load or create config
    if args.config:
        with open(args.config) as f:
            config_dict = json.load(f)
        config = TrainingConfig(**config_dict)
    else:
        config = TrainingConfig()

    # Override with CLI args
    if args.dataset:
        config.train_dataset = args.dataset
    if args.output:
        config.output_dir = args.output
    if args.model:
        config.model_name = args.model
    if args.max_steps:
        config.max_steps = args.max_steps

    # Train or test
    if args.test:
        test_prompts = [
            ALPACA_PROMPT.format(
                "type Indexer { id: ID!, totalStake: BigInt, selfStake: BigInt }",
                "What is the total stake of all indexers?",
                ""
            )
        ]
        test_inference(config.output_dir, test_prompts)
    else:
        train(config)


if __name__ == "__main__":
    main()
