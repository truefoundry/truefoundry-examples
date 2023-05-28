import copy
import gc
import json
import logging
import math
import os
import shutil
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import mlfoundry
import numpy as np
import torch
from cloudfiles import CloudFile
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
    HfArgumentParser,
    TrainerCallback,
    pipeline,
)
from transformers.integrations import rewrite_logs
from transformers.utils import is_torch_bf16_available, is_torch_tf32_available

logging.basicConfig(level=logging.INFO)
IGNORE_INDEX = -100  # -100 is the default ignore index in CrossEntropyLoss
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"
mlfoundry_client = mlfoundry.get_client()


@dataclass
class OtherArguments:
    model_id: str = field(metadata={"help": "Huggingface hub model ID"})
    train_data: str = field(metadata={"help": "URL to the jsonl training dataset"})
    ml_repo: str = field(metadata={"help": "ML Repo to put the model to"})
    eval_size: Optional[float] = field(
        default=0.1,
        metadata={"help": "Proportion of training data to use as evaluation set. Ignored if `eval_data` is passed"}
    )
    eval_data: Optional[str] = field(
        default=None,
        metadata={"help": "URL to the jsonl evaluation dataset. Overrides eval_size"}
    )
    checkpoint_artifact_name: str = field(
        default=None,
        metadata={
            "help": "ML Repo artifact name to save checkpoints. \n"
                    "The artifact will be created if it does not exist under the give ML Repo"
        },
    )
    max_length: Optional[int] = field(
        default=None,
        metadata={
            "help": "Max length to truncate the examples to. By default we try to pick "
                    "from tokenizer config (default: None)"
        },
    )
    max_num_samples: Optional[int] = field(
        default=None,
        metadata={"help": "For quick debugging purposes, how many samples to use (default: all)"}
    )


# --- Model checkpointing and logging utils ---

def resolve_checkpoint_artifact_name(
        checkpoint_artifact_name: Optional[str],
) -> Optional[str]:
    if checkpoint_artifact_name:
        return checkpoint_artifact_name
    if os.getenv("TFY_INTERNAL_JOB_RUN_NAME"):
        job_name = os.getenv("TFY_INTERNAL_JOB_RUN_NAME")
        return f"checkpoint-{job_name}"
    return None


class Callback(TrainerCallback):
    def __init__(
            self,
            run: mlfoundry.MlFoundryRun,
            checkpoint_artifact_name: Optional[str] = None,
    ):
        self._mlf_run = run
        self._checkpoint_artifact_name = checkpoint_artifact_name

        if not self._checkpoint_artifact_name:
            logging.warning(
                "checkpoint_artifact_name not passed. Checkpoints will not be logged to MLFoundry"
            )

    def on_evaluate(self, args, state, control, metrics, **kwargs):
        # TODO (chiragjn): Hack for now, needs to be moved to `compute_metrics`
        if "eval_loss" in metrics:
            try:
                metrics["eval_perplexity"] = math.exp(metrics["eval_loss"])
            except OverflowError:
                metrics["eval_perplexity"] = float("inf")

    def on_log(self, args, state, control, logs, model=None, **kwargs):
        if state.is_world_process_zero:
            metrics = {}
            for k, v in logs.items():
                if isinstance(v, (int, float, np.integer, np.floating)):
                    metrics[k] = v
                else:
                    logging.warning(
                        f'Trainer is attempting to log a value of "{v}" of'
                        f' type {type(v)} for key "{k}" as a metric.'
                        " MLflow's log_metric() only accepts float and"
                        " int types so we dropped this attribute."
                    )
            self._mlf_run.log_metrics(rewrite_logs(metrics), step=state.global_step)

    def on_save(self, args, state, control, **kwargs):
        if state.is_world_process_zero and self._checkpoint_artifact_name:
            ckpt_dir = f"checkpoint-{state.global_step}"
            artifact_path = os.path.join(args.output_dir, ckpt_dir)
            description = None
            if os.getenv("TFY_INTERNAL_COMPONENT_NAME"):
                description = (
                    f"Checkpoint from finetuning job={os.getenv('TFY_INTERNAL_COMPONENT_NAME')}"
                    f" run={os.getenv('TFY_INTERNAL_JOB_RUN_NAME')}"
                )
            self._mlf_run.log_artifact(
                name=self._checkpoint_artifact_name,
                artifact_paths=[(artifact_path,)],
                step=state.global_step,
                description=description,
            )


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


# --- Data Processing Utils ---

class TuneDataset(Dataset):
    """Dead simple torch dataset wrapper. Attention masks are created in collator"""

    def __init__(self, input_ids, labels):
        self.input_ids = input_ids
        self.labels = labels

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i):
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])


class DatasetBuilder:
    """Dataset agnostic class to take in input_ids and labels and spit out tokens"""

    def __init__(self, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def batch_tokenize(self, texts):
        """Tokenizes text. Presently doesn't pad inputs, just returns input ids."""
        tokenized = [
            self.tokenizer(
                prompt,
                return_tensors="pt",
                padding="longest",
                max_length=self.max_length,
                truncation=True
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

    def __init__(self, tokenizer, max_length, train_on_prompt=True):
        super().__init__(tokenizer, max_length)
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
            labels = [
                self.pad_to_multiple(val, IGNORE_INDEX)
                for val in labels
            ]

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


def load_data(path, max_num_samples: Optional[int] = None):
    raw_data = CloudFile(path).get().decode("utf-8").split("\n")
    data = []
    n = max_num_samples if max_num_samples else len(raw_data)
    for i in range(n):
        line = raw_data[i]
        try:
            json_object = json.loads(line)
        except json.decoder.JSONDecodeError:
            pass
        else:
            data.append(json_object)
    return data


# --- Core Training Code ---


def download_last_checkpoint_if_present(
        run: mlfoundry.MlFoundryRun, checkpoint_artifact_name: str, local_dir: str
) -> Optional[str]:
    # TODO: update this once we have better API
    checkpoint_artifact_fqn = (
        f"artifact:{'/'.join(run.fqn.split('/')[:2])}/{checkpoint_artifact_name}"
    )
    try:
        # TODO (chiragjn): We can use `:latest` tag
        latest_checkpoint_artifact = next(
            mlfoundry_client.list_artifact_versions(checkpoint_artifact_fqn)
        )
    except StopIteration:
        logging.info(
            "No previous checkpoints found at artifact=%r", checkpoint_artifact_fqn
        )
        return
    # TODO: We should have specific exception to identify if the artifact
    #   does not exist
    except Exception as ex:
        logging.info("No previous checkpoints found. Message=%s", ex)
        return

    logging.info(
        "Downloading last checkpoint from artifact version=%r step=%r to resume training",
        latest_checkpoint_artifact.fqn,
        latest_checkpoint_artifact.step,
    )
    os.makedirs(local_dir, exist_ok=True)
    local_dir = os.path.join(local_dir, f"checkpoint-{latest_checkpoint_artifact.step}")
    with tempfile.TemporaryDirectory() as temp_dir:
        path = latest_checkpoint_artifact.download(temp_dir)
        shutil.move(path, local_dir)
    return local_dir


def train(
        *,
        run: mlfoundry.MlFoundryRun,
        training_arguments: TrainingArguments,
        other_arguments: OtherArguments,
):
    set_seed(training_arguments.seed)

    logging.info(f"Loading train dataset {other_arguments.train_data}...")
    train_data = load_data(other_arguments.train_data, max_num_samples=other_arguments.max_num_samples)
    eval_data = other_arguments.eval_data
    if eval_data:
        logging.info(f"Loading eval dataset {other_arguments.eval_data}...")
        eval_data = load_data(train_data, max_num_samples=other_arguments.max_num_samples)
    elif other_arguments.eval_size:
        logging.info(f"No eval dataset given, splitting from training dataset...")
        train_data, eval_data = train_test_split(
            train_data,
            test_size=other_arguments.eval_size,
            random_state=training_arguments.data_seed
        )

    logging.info("Checking for any past checkpoints...")
    checkpoint_artifact_name = resolve_checkpoint_artifact_name(
        other_arguments.checkpoint_artifact_name
    )
    if checkpoint_artifact_name:
        last_checkpoint_dir = download_last_checkpoint_if_present(
            run,
            checkpoint_artifact_name=checkpoint_artifact_name,
            local_dir=training_arguments.output_dir,
        )
    else:
        last_checkpoint_dir = None

    load_from = last_checkpoint_dir if last_checkpoint_dir else other_arguments.model_id

    logging.info("Loading model...")
    # TODO (chiragjn): Should we pass a torch_dtype here?
    model = AutoModelForCausalLM.from_pretrained(
        load_from,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        use_cache=False if training_arguments.gradient_checkpointing else True,
    )

    logging.info("Loading tokenizer...")
    try:
        # Note: First we try loading with use_fast=False because for some models conversion takes too long
        tokenizer = AutoTokenizer.from_pretrained(
            load_from,
            trust_remote_code=True,
            use_fast=False
        )
    except ValueError:
        tokenizer = AutoTokenizer.from_pretrained(
            load_from,
            trust_remote_code=True,
        )
    special_tokens_dict = {}
    if tokenizer.pad_token is None:
        logging.info("Pad token missing, adding a pad token")
        special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
    if tokenizer.eos_token is None:
        logging.info("EOS token missing, adding a EOS token")
        special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
    if tokenizer.bos_token is None:
        logging.info("BOS token missing, adding a BOS token")
        special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
    if tokenizer.unk_token is None:
        logging.info("UNK token missing, adding a UNK token")
        special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN
    tokenizer.add_special_tokens(special_tokens_dict)
    # TODO (chiragjn): Consider adding fake tokens to vocab to pad to multiple of 64. Can provide better throughput
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    if num_new_tokens > 0:
        model.resize_token_embeddings(len(tokenizer))
        # There are some strategies that also assign unk token as pad token
        # We can also assign the average of all embeddings here for new tokens that got added

    max_length = other_arguments.max_length
    if max_length is None:
        if tokenizer.model_max_length > int(1e6):
            logging.info(f"tokenizer config does not have proper model_max_length set. Looking at model config")
            for length_setting in ["max_sequence_length", "n_positions", "max_position_embeddings"]:
                max_length = getattr(model.config, length_setting, None)
                if max_length:
                    logging.info(f"Assuming value of {length_setting} from model config as max length: {max_length}")
                    break
            if not max_length:
                logging.info(f"Found no max length setting, falling back to default of 512")
                max_length = 512
        else:
            max_length = tokenizer.model_max_length
    logging.info(f"Finally using max_length: {max_length}")

    logging.info("Building dataset...")
    dataset_builder = CausalDatasetBuilder(tokenizer=tokenizer, max_length=max_length)
    train_dataset = dataset_builder.construct_dataset(train_data)
    logging.info(f"Train data size: {len(train_dataset)}")
    if eval_data:
        eval_dataset = dataset_builder.construct_dataset(eval_data)
        logging.info(f"Eval data size: {len(eval_dataset)}")
    else:
        eval_dataset = None

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    logging.info("Training...")
    # TODO (chiragjn): Add text generation metrics to `compute_metrics`
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=training_arguments,
        data_collator=SequenceDataCollator(tokenizer, multiple_of=8),
        # Tensor cores are used when tensor dims are multiple of 8
    )
    trainer.add_callback(
        Callback(run, checkpoint_artifact_name=checkpoint_artifact_name)
    )
    trainer.train(resume_from_checkpoint=last_checkpoint_dir)
    trainer.save_model(output_dir=training_arguments.output_dir)


def filter_trainer_args_for_logging(trainer_args: TrainingArguments) -> Dict[str, Any]:
    # TODO (chiragjn): Update this list
    return {
        "num_train_epochs": trainer_args.num_train_epochs,
        "per_device_train_batch_size": trainer_args.per_device_train_batch_size,
        "learning_rate": trainer_args.learning_rate,
        "lr_scheduler_type": trainer_args.lr_scheduler_type,
        "weight_decay": trainer_args.weight_decay,
        "max_grad_norm": trainer_args.max_grad_norm,
        "gradient_accumulation_steps": trainer_args.gradient_accumulation_steps,
        "warmup_ratio": trainer_args.warmup_ratio,
    }


def main():
    parser = HfArgumentParser(
        (TrainingArguments, OtherArguments), description="Fine-tune a language model on a text dataset"
    )
    training_arguments, other_arguments = parser.parse_args_into_dataclasses()
    # TODO (chiragjn): Think about the optimal value for half_precision_backend
    training_arguments.bf16 = is_torch_bf16_available()
    training_arguments.tf32 = is_torch_tf32_available()
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S")
    *_, model_name = other_arguments.model_id.rsplit("/", 1)
    model_name = model_name.replace(".", "-")
    model_name = f"{model_name}-{timestamp}"

    with mlfoundry_client.create_run(
            ml_repo=other_arguments.ml_repo, run_name=f"finetune-{timestamp}"
    ) as run:
        run.log_params(other_arguments, flatten_params=True)
        run.log_params(
            filter_trainer_args_for_logging(training_arguments), flatten_params=True
        )
        # TODO: there are 110 params in training_arguments, we do not need to log all of them.
        # run.log_params(training_arguments.to_sanitized_dict(), flatten_params=True)
        train(run=run, training_arguments=training_arguments, other_arguments=other_arguments)
        save_model(
            run=run, training_arguments=training_arguments, model_name=model_name
        )


if __name__ == "__main__":
    main()


# --deepspeed ./ds_z3_config.json
# --gradient_checkpointing