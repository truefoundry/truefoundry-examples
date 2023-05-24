import copy
import gc
import json
from datetime import datetime, timezone

import mlfoundry
import torch
from cloudfiles import CloudFile
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    AutoModelForCausalLM,
    HfArgumentParser,
)
from transformers import pipeline

# TODO (chiragjn): Add support for other task types

torch.manual_seed(42)
IGNORE_INDEX = -100  # TODO (chiragjn): Eliminate this magic number


class DatasetBuilder:
    """Dataset agnostic class to take in input_ids and labels and spit out tokens"""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def batch_tokenize(self, texts):
        """Tokenizes text. Presently doesn't pad inputs, just returns input ids."""
        tokenized = [
            self.tokenizer(
                prompt, return_tensors="pt", padding="longest", truncation=True
            ).input_ids
            for prompt in texts
        ]
        return tokenized

    def construct_dataset(self, input_data):
        prompts = [val["prompt"] for val in input_data]
        tokenized_input_ids = self.batch_tokenize(prompts)
        labels = [val["completion"] for val in input_data]
        tokenized_labels = self.batch_tokenize(labels)
        return TuneDataset(tokenized_input_ids, tokenized_labels)


class CausalDatasetBuilder(DatasetBuilder):
    """Builds generative dataset for Causal LM."""

    def __init__(self, tokenizer, train_on_prompt=True):
        super().__init__(tokenizer)
        self.train_on_prompt = train_on_prompt

    def construct_dataset(self, input_data):
        labels = [
            val["prompt"] + "\n" + val["completion"] + self.tokenizer.eos_token
            for val in input_data
        ]
        input_ids = [val.squeeze() for val in self.batch_tokenize(labels)]
        labels = copy.deepcopy(input_ids)
        if self.train_on_prompt:
            return TuneDataset(input_ids, labels)
        # masking prompt
        prompts = [val["prompt"] for val in input_data]
        tokenized_prompts = self.batch_tokenize(prompts)
        prompt_lens = [val.shape[1] for val in tokenized_prompts]

        for label, source_len in zip(labels, prompt_lens):
            label[:source_len] = IGNORE_INDEX
        return TuneDataset(input_ids, labels)


class TuneDataset(Dataset):
    """Dead simple torch dataset wrapper. Attention masks are created in collator"""

    def __init__(self, input_ids, labels):
        self.input_ids = input_ids
        self.labels = labels

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i):
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])


class SequenceDataCollator:
    """Collate examples for dynamic batch construction in supervised fine-tuning."""

    def __init__(self, tokenizer, multiple_of=None):
        self.tokenizer = tokenizer
        self.multiple_of = multiple_of
        self.cache_count = 0

    def pad_to_multiple(self, tensor, value):
        # taking advantage of tensor cores, perhaps
        multiple = self.multiple_of
        target_length = (tensor.size(0) + multiple - 1) // multiple * multiple
        return torch.nn.functional.pad(
            tensor, (0, target_length - tensor.size(0)), value=value
        )

    def __call__(self, instances):
        input_ids, labels = tuple(
            [instance[key] for instance in instances] for key in ("input_ids", "labels")
        )
        if self.multiple_of:
            input_ids = [
                self.pad_to_multiple(val, self.tokenizer.pad_token_id)
                for val in input_ids
            ]
            labels = [self.pad_to_multiple(val, IGNORE_INDEX) for val in labels]

        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX
        )  # -100 tells torch to ignore these tokens in loss computation.

        if self.cache_count < 1:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            self.cache_count += 1

        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )


def load_data(path, num_samples: int = -1):
    raw_data = CloudFile(path).get().decode("utf8").split("\n")
    data = []
    n = num_samples if num_samples >= 0 else len(raw_data)
    for i in range(n):
        line = raw_data[i]
        try:
            json_object = json.loads(line)
        except json.decoder.JSONDecodeError:
            pass
        else:
            data.append(json_object)
    return data


def save_model(
    run: mlfoundry.MlFoundryRun,
    training_arguments: TrainingArguments,
    model_name: str,
):
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("Saving Model...")
    p = pipeline(
        "text-generation",
        model=training_arguments.output_dir,
        tokenizer=training_arguments.output_dir,
    )
    run.log_model(
        name=model_name,
        model=p,
        framework="transformers",
        metadata=training_arguments.to_sanitized_dict(),
    )


def train(
    model_id: str,
    train_data: str,
    training_arguments: TrainingArguments,
    num_samples: int = -1,
    **kwargs,
):
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        model.resize_token_embeddings(len(tokenizer))

    print(f"Loading dataset {train_data}...")
    train_data = load_data(train_data, num_samples=num_samples)

    print("Building dataset...")
    p = CausalDatasetBuilder(tokenizer)
    train_dataset = p.construct_dataset(train_data)
    eval_dataset = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("Training...")
    # TODO (chiragjn): Add metrics and evaluations?
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=training_arguments,
        data_collator=SequenceDataCollator(tokenizer, 8),  # depends on bf16 value
        # TODO (chiragjn): Eliminate this magic number 8
    )
    trainer.train()
    trainer.save_model(output_dir=training_arguments.output_dir)


def main():
    parser = HfArgumentParser(
        TrainingArguments, description="Fine-tune a language model on a text dataset"
    )
    parser.add_argument(
        "--model_id", type=str, required=True, help="Huggingface hub model ID"
    )
    parser.add_argument(
        "--train_data", type=str, required=True, help="Path to the json dataset"
    )
    parser.add_argument(
        "--ml_repo", type=str, required=True, help="ML Repo to put the model to"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=-1,
        help="How many samples to use (default: all)",
    )
    training_arguments, other_args = parser.parse_args_into_dataclasses()

    client = mlfoundry.get_client()
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S")

    *_, model_name = other_args.model_id.rsplit("/", 1)
    model_name = model_name.replace(".", "-")
    model_name = f"{model_name}-{timestamp}"

    with client.create_run(
        ml_repo=other_args.ml_repo, run_name=f"finetune-{timestamp}"
    ) as run:
        train(training_arguments=training_arguments, **vars(other_args))
        save_model(run=run, training_arguments=training_arguments, model_name=model_name)


if __name__ == "__main__":
    main()
