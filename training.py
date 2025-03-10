import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

from clearml import Task
from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling
import torch
from transformers import AutoTokenizer
from datasets import load_dataset

from EmbeddingsDivision import EmbeddingsDivision

from accelerate import Accelerator

os.environ["WANDB_DISABLED"] = "true"

print(torch.cuda.device_count())
print(torch.cuda.is_available())

model_id = "meta-llama/Llama-3.2-1B-Instruct"

task = Task.init(project_name='Paper', task_name='Experiment')

def tokenize_function_maker(tokenizer):
    def inner(examples):
        return {'input_ids': tokenizer.apply_chat_template(examples["text"], tokenize=True, padding='max_length', truncation=True, max_length=4096)}
    return inner


def main():
    tokenizer = AutoTokenizer.from_pretrained("ikkiren/research_tokenizer")
    tokenizer.pad_token = tokenizer.eos_token

    ds = load_dataset("ikkiren/merged_instruct_refactor",
                      token="")

    ds_train = ds["train"]
    ds_test = ds["test"]

    ds_train = ds_train.map(tokenize_function_maker(
        tokenizer), num_proc=20, remove_columns=["text"], batched=True)
    ds_test = ds_test.map(tokenize_function_maker(
        tokenizer), num_proc=20, remove_columns=["text"], batched=True)

    model_id = "meta-llama/Llama-3.2-1B-Instruct"

    model = EmbeddingsDivision(model_id, device='auto',)
    model.divide_embeddings(0.667641)
    model = model.model

    for name, param in model.named_parameters():
        if "embed_tokens2" in name:
            param.requires_grad = True
            print(name)
        else:
            param.requires_grad = False

    training_args = TrainingArguments(
        # auto_find_batch_size=True,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=1,
        do_train=True,
        do_eval=False,

        # accelerator_config=accelerator,
        
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),

        num_train_epochs=1,

        learning_rate=5e-05,
        log_level="info",
        logging_steps=1,
        logging_strategy="steps",
        lr_scheduler_type="constant",
        
        save_steps=20000,
        save_total_limit=1,    
    )

    accelerator = Accelerator()
    trainer = accelerator.prepare(Trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=ds_train,
        eval_dataset=ds_test,
        args=training_args,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
    ))
    
    trainer.train()


    model.save_pretrained("/research/model")

    model.push_to_hub("research")


if __name__ == '__main__':
    main()
