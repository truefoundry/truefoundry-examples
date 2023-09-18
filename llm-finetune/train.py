import copy
import gc
import json
import logging
import math
import os
import re
import shutil
import sys
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import mlfoundry
import numpy as np
import torch
import torch.backends.cuda
import torch.distributed
from cloudfiles import CloudFile
from datasets import Dataset, DatasetDict
from deepspeed.utils.zero_to_fp32 import convert_zero_checkpoint_to_fp32_state_dict
from huggingface_hub import scan_cache_dir
from peft import AutoPeftModelForCausalLM, LoraConfig, get_peft_model
from sklearn.model_selection import train_test_split
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    IntervalStrategy,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    pipeline,
    set_seed,
)
from transformers.deepspeed import is_deepspeed_zero3_enabled
from transformers.integrations import rewrite_logs
from transformers.utils import (
    WEIGHTS_NAME,
    is_torch_bf16_gpu_available,
    is_torch_tf32_available,
)
from transformers.utils import logging as hf_logging_utils

# TODO (chiragjn):
#   - Test resume from checkpoint for both full and lora
#   - Add support for 8 bit and 4 bit QLora
#   - Test support for fp16 on older GPUs
#   - Write a script to automatically capture gpu and memory metrics with different configurations
#   - Test and fix Deepspeed weight gathering bugs during checkpointing if any
#   - Test and fix code saving for models that have custom code
#   - Add support to push to HF Hub, as well as ability to read gated models
#   - Add support for multi gpu Lora
#   - Add support for text2text-generation
#   - Add support to use  Apex FusedAdam

TFY_INTERNAL_JOB_NAME = os.getenv("TFY_INTERNAL_COMPONENT_NAME")
TFY_INTERNAL_JOB_RUN_NAME = os.getenv("TFY_INTERNAL_JOB_RUN_NAME")
THIS_DIR = os.path.abspath(os.path.dirname(__name__))
CACHE_DIR = os.path.join(THIS_DIR, ".cache")
EXPORT_ZERO3_CHECKPOINT_TO_FP32 = False
logger = logging.getLogger(__name__)

IGNORE_INDEX = -100  # -100 is the default ignore index in CrossEntropyLoss
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"


@dataclass
class HFTrainingArguments(TrainingArguments):
    def __post_init__(self):
        if not self.fp16:
            self.bf16 = not self.no_cuda and torch.cuda.is_available() and is_torch_bf16_gpu_available()
            self.tf32 = not self.no_cuda and torch.cuda.is_available() and is_torch_tf32_available()
        if self.save_strategy == IntervalStrategy.NO:
            self.load_best_model_at_end = False
        super().__post_init__()


@dataclass
class OtherArguments:
    model_id: str = field(metadata={"help": "Huggingface hub model ID"})
    # TODO (chiragjn): Make this optional, because now we have --report_to_mlfoundry
    ml_repo: str = field(metadata={"help": "ML Repo to put the model to"})
    train_data: str = field(metadata={"help": "URL to the jsonl training dataset"})
    eval_size: Optional[float] = field(
        default=0.1,
        metadata={"help": "Proportion of training data to use as evaluation set. Ignored if `eval_data` is passed"},
    )
    eval_data: Optional[str] = field(
        default="NA",
        metadata={"help": "URL to the jsonl evaluation dataset. Overrides eval_size. Leave as NA if not available"},
    )
    report_to_mlfoundry: bool = field(
        default=True,
        metadata={"help": "Use mlfoundry to log metrics, checkpoints and model"},
    )
    log_checkpoints_to_mlfoundry: bool = field(
        default=True,
        metadata={"help": "If to log intermediate checkpoints to mlfoundry"},
    )
    # TODO (chiragjn): Add option to control max shard size
    checkpoint_artifact_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "ML Repo artifact name to save checkpoints. \n"
            "The artifact will be created if it does not exist under the give ML Repo"
        },
    )
    use_lora: bool = field(
        default=False,
        metadata={"help": "If to train the model with LoRa"},
    )
    #  "c_attn", "down_proj", "gate_proj", "up_proj", "query_key_value" "dense", "dense_h_to_4h", "dense_4h_to_h",
    lora_config: str = field(
        default=json.dumps(
            dict(
                r=8,
                lora_alpha=32,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM",
            )
        ),
        metadata={"help": "Json encoded string containing config for lora training"},
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
        metadata={"help": "For quick debugging purposes, how many samples to use (default: all)"},
    )
    cleanup_output_dir_on_start: bool = field(
        default=False,
        metadata={"help": "Cleanup output dir at the start of training run"},
    )


def get_torch_dtype(training_arguments: HFTrainingArguments):
    torch_dtype = None
    if training_arguments.bf16:
        torch_dtype = torch.bfloat16
    elif training_arguments.fp16:
        torch_dtype = torch.float16
    return torch_dtype


# --- Model checkpointing and logging utils ---


def resolve_checkpoint_artifact_name(
    checkpoint_artifact_name: Optional[str],
) -> Optional[str]:
    if checkpoint_artifact_name:
        return checkpoint_artifact_name
    if TFY_INTERNAL_JOB_RUN_NAME:
        job_name = TFY_INTERNAL_JOB_RUN_NAME
        return f"checkpoint-{job_name}"
    return None


def download_last_checkpoint_if_present(
    run: mlfoundry.MlFoundryRun, checkpoint_artifact_name: str, local_dir: str
) -> Optional[str]:
    mlfoundry_client = mlfoundry.get_client()
    try:
        # TODO (chiragjn): We can use `:latest` tag
        latest_checkpoint_artifact = next(
            mlfoundry_client.list_artifact_versions(ml_repo=run.ml_repo, name=checkpoint_artifact_name)
        )
    except StopIteration:
        logger.info(
            f"No previous checkpoints found at artifact={checkpoint_artifact_name!r} in run={run.ml_repo!r}",
        )
        return
    # TODO: We should have specific exception to identify if the artifact
    #   does not exist
    except Exception as ex:
        logger.info("No previous checkpoints found. Message=%s", ex)
        return

    logger.info(
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


def get_checkpoint_for_resume_if_any(
    training_arguments: HFTrainingArguments,
    run: Optional[mlfoundry.MlFoundryRun],
    checkpoint_artifact_name: Optional[str] = None,
) -> Optional[str]:
    last_checkpoint_info_path = os.path.join(CACHE_DIR, "last_checkpoint_info.json")
    last_checkpoint_dir = None
    if training_arguments.local_rank <= 0:
        if run:
            logger.info("Checking for any past checkpoints...")
            if checkpoint_artifact_name:
                last_checkpoint_dir = download_last_checkpoint_if_present(
                    run,
                    checkpoint_artifact_name=checkpoint_artifact_name,
                    local_dir=training_arguments.output_dir,
                )
        with open(last_checkpoint_info_path, "w") as f:
            last_checkpoint_info = {"last_checkpoint_dir": last_checkpoint_dir}
            json.dump(last_checkpoint_info, f)
    else:
        with open(last_checkpoint_info_path, "r") as f:
            last_checkpoint_info = json.load(f)
        last_checkpoint_dir = last_checkpoint_info["last_checkpoint_dir"]
    return last_checkpoint_dir


def cleanup_checkpoints(
    training_arguments: HFTrainingArguments,
):
    logger.info("Cleaning up older checkpoints...")
    for f in os.listdir(training_arguments.output_dir):
        f_path = os.path.join(training_arguments.output_dir, f)
        if os.path.isdir(f_path) and f.startswith("checkpoint-"):
            shutil.rmtree(f_path)


def log_model_as_pipeline(
    run: mlfoundry.MlFoundryRun, training_arguments: HFTrainingArguments, model_name: str, hf_hub_model_id: str
):
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.info("Saving Model...")
    cleanup_checkpoints(training_arguments=training_arguments)

    hf_cache_info = scan_cache_dir()
    files_to_save = []
    for repo in hf_cache_info.repos:
        if repo.repo_id == hf_hub_model_id:
            for revision in repo.revisions:
                for file in revision.files:
                    if file.file_path.name.endswith(".py"):
                        files_to_save.append(file.file_path)
                break

    additional_files = []
    # copy the files to output_dir of pipeline
    for file_path in files_to_save:
        match = re.match(r".*snapshots\/[^\/]+\/(.*)", str(file_path))
        if match:
            relative_path = match.group(1)
            destination_path = os.path.join(training_arguments.output_dir, relative_path)
            os.makedirs(os.path.dirname(destination_path), exist_ok=True)
            shutil.copy(str(file_path), destination_path)
            additional_files.append((destination_path, os.path.join("model", "pipeline", relative_path)))
        else:
            logger.warning("Python file in hf model cache in unknown path:", file_path)

    p = pipeline(
        "text-generation",
        model=training_arguments.output_dir,
        tokenizer=training_arguments.output_dir,
        trust_remote_code=True,
        torch_dtype=get_torch_dtype(training_arguments=training_arguments),
        device_map="auto",  # We load on GPUs if available because we can be low on regular memory
    )
    run.log_model(
        name=model_name,
        model=p,
        framework="transformers",
        metadata=training_arguments.to_sanitized_dict(),
        additional_files=additional_files,
    )


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


class Callback(TrainerCallback):
    def __init__(
        self,
        run: Optional[mlfoundry.MlFoundryRun] = None,
        checkpoint_artifact_name: Optional[str] = None,
        log_checkpoints_to_mlfoundry: bool = True,
    ):
        self._run = run
        self._checkpoint_artifact_name = checkpoint_artifact_name
        self._log_checkpoints_to_mlfoundry = log_checkpoints_to_mlfoundry

        if not self._checkpoint_artifact_name:
            logger.warning("checkpoint_artifact_name not passed. Checkpoints will not be logged to MLFoundry")

    # noinspection PyMethodOverriding
    def on_log(self, args, state, control, logs, model=None, **kwargs):
        # TODO (chiragjn): Hack for now, needs to be moved to `compute_metrics`
        #   unfortunately compute metrics does not give us already computed metrics like eval_loss
        if not state.is_world_process_zero:
            return

        for loss_key, perplexity_key in [("loss", "train_perplexity"), ("eval_loss", "eval_perplexity")]:
            if loss_key in logs:
                try:
                    perplexity = math.exp(logs[loss_key])
                except OverflowError:
                    perplexity = float("inf")
                    logger.warning(f"Encountered inf in eval perplexity, cannot log it as a metric")
                logger.info(f"{perplexity_key}: {perplexity}")
                logs[perplexity_key] = perplexity

        logger.info(f"Metrics: {logs}")
        if not self._run:
            return

        metrics = {}
        for k, v in logs.items():
            if isinstance(v, (int, float, np.integer, np.floating)) and math.isfinite(v):
                metrics[k] = v
            else:
                logger.warning(
                    f'Trainer is attempting to log a value of "{v}" of'
                    f' type {type(v)} for key "{k}" as a metric.'
                    " Mlfoundry's log_metric() only accepts finite float and"
                    " int types so we dropped this attribute."
                )
        self._run.log_metrics(rewrite_logs(metrics), step=state.global_step)

    def on_save(self, args, state, control, **kwargs):
        if not state.is_world_process_zero:
            return

        if not self._run or not self._checkpoint_artifact_name:
            return

        if not self._log_checkpoints_to_mlfoundry:
            return

        ckpt_dir = f"checkpoint-{state.global_step}"
        artifact_path = os.path.join(args.output_dir, ckpt_dir)
        description = None
        if TFY_INTERNAL_JOB_NAME:
            description = f"Checkpoint from finetuning job={TFY_INTERNAL_JOB_NAME} run={TFY_INTERNAL_JOB_RUN_NAME}"
        self._run.log_artifact(
            name=self._checkpoint_artifact_name,
            artifact_paths=[(artifact_path,)],
            step=state.global_step,
            description=description,
        )


# --- Data Processing Utils ---


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
                truncation=True,
            ).input_ids
            for prompt in texts
        ]
        return tokenized

    def construct_dataset(self, input_batch):
        tokenized_input_ids = self.batch_tokenize(input_batch["prompt"])
        tokenized_labels = self.batch_tokenize(input_batch["completion"])
        return {"input_ids": tokenized_input_ids, "labels": tokenized_labels}


class CausalDatasetBuilder(DatasetBuilder):
    """Builds generative dataset for Causal LM."""

    def __init__(self, tokenizer, max_length, train_on_prompt=True):
        super().__init__(tokenizer, max_length)
        self.train_on_prompt = train_on_prompt

    def construct_dataset(self, input_batch):
        labels = []
        for prompt, completion in zip(input_batch["prompt"], input_batch["completion"]):
            labels.append(prompt + "\n" + completion + self.tokenizer.eos_token)
        input_ids = [val.squeeze() for val in self.batch_tokenize(labels)]
        labels = copy.deepcopy(input_ids)
        if not self.train_on_prompt:
            tokenized_prompts = self.batch_tokenize(input_batch["prompt"])
            prompt_lens = [val.shape[1] for val in tokenized_prompts]
            for label, source_len in zip(labels, prompt_lens):
                label[:source_len] = IGNORE_INDEX
        return {"input_ids": input_ids, "labels": labels}


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
        return torch.nn.functional.pad(tensor, (0, target_length - tensor.size(0)), value=value)

    def __call__(self, instances):
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        if self.multiple_of:
            input_ids = [self.pad_to_multiple(val, self.tokenizer.pad_token_id) for val in input_ids]
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


def _read_lines_from_files(download_path):
    for root, dirs, files in os.walk(download_path):
        for file in files:
            filepath = os.path.join(root, file)
            filename = os.path.basename(filepath)
            if filename.endswith(".jsonl") and not filename.startswith("."):
                logger.info(f"Loading file {filename} ...")
                with open(filepath) as f:
                    for line in f.readlines():
                        yield line


def _read_lines_from_cloudfile(path):
    raw_data = CloudFile(path).get().decode("utf-8").split("\n")
    for line in raw_data:
        yield line


def load_data(path, max_num_samples: Optional[int] = None):
    data = []
    n = max_num_samples if max_num_samples else -1
    count = 0
    with tempfile.TemporaryDirectory() as download_dir:
        if path.startswith("mlfoundry://") or path.startswith("artifact:"):
            logger.info("Downloading artifact from mlfoundry")
            client = mlfoundry.get_client()
            if path.startswith("mlfoundry://"):
                _, artifact_version_fqn = path.split("mlfoundry://", 1)
            else:
                artifact_version_fqn = path
            download_path = client.get_artifact(artifact_version_fqn).download(download_dir)
            lines = _read_lines_from_files(download_path)
        else:
            logger.info(f"Loading data from link: {path}")
            lines = _read_lines_from_cloudfile(path)
        for line in lines:
            if n > 0 and count >= n:
                break
            try:
                json_object = json.loads(line)
            except json.decoder.JSONDecodeError:
                pass
            else:
                data.append(json_object)
                count += 1
    return data


def get_data(training_arguments: HFTrainingArguments, other_arguments: OtherArguments):
    train_data, eval_data = None, None
    if training_arguments.local_rank <= 0:
        logger.info(f"Loading train dataset ...")
        train_data = load_data(other_arguments.train_data, max_num_samples=other_arguments.max_num_samples)
        eval_data = other_arguments.eval_data
        if eval_data and eval_data != "NA":
            logger.info(f"Loading eval dataset {other_arguments.eval_data}...")
            eval_data = load_data(eval_data, max_num_samples=other_arguments.max_num_samples)
        elif other_arguments.eval_size:
            logger.info(f"No eval dataset given, splitting from training dataset...")
            train_data, eval_data = train_test_split(
                train_data,
                test_size=other_arguments.eval_size,
                random_state=training_arguments.data_seed,
            )
    return train_data, eval_data


def build_dataset(train_data, eval_data, tokenizer, max_length, training_arguments):
    logger.info("Building dataset...")
    dataset_cache_path = os.path.join(CACHE_DIR, "dataset")
    if training_arguments.local_rank <= 0:
        builder = CausalDatasetBuilder(tokenizer=tokenizer, max_length=max_length)
        dataset_dict = DatasetDict(train=Dataset.from_list(train_data), eval=Dataset.from_list(eval_data))
        dataset_dict = dataset_dict.map(
            builder.construct_dataset,
            remove_columns=["prompt", "completion"],
            batched=True,
            batch_size=32,
        )
        dataset_dict.save_to_disk(dataset_cache_path)
    else:
        logger.info("Loading datasets from cache ...")
        dataset_dict = DatasetDict.load_from_disk(dataset_cache_path)
    dataset_dict = dataset_dict.with_format("torch")
    train_dataset, eval_dataset = dataset_dict["train"], dataset_dict["eval"]
    logger.info(f"Train data size: {len(train_dataset)}")
    logger.info(f"Eval data size: {len(eval_dataset)}")
    return train_dataset, eval_dataset


# --- Core Training Code ---


def setup(training_arguments: HFTrainingArguments):
    global logger
    os.makedirs(CACHE_DIR, exist_ok=True)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter(fmt=f"%(asctime)s [Rank-{training_arguments.local_rank}] %(levelname)s %(message)s")
    handler.setFormatter(formatter)

    logger.setLevel(logging.DEBUG)
    logger.addHandler(handler)

    hf_logging_utils.disable_default_handler()
    hf_logging_utils.add_handler(handler)


def get_model(model_source: str, training_arguments: HFTrainingArguments):
    # TODO (chiragjn): Should we pass a torch_dtype here?
    logger.info("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_source,
        trust_remote_code=True,
        use_cache=False if training_arguments.gradient_checkpointing else True,
        torch_dtype=get_torch_dtype(training_arguments),
    )
    if training_arguments.gradient_checkpointing:
        model.config.use_cache = False
    return model


def get_tokenizer(model_source: str):
    logger.info("Loading tokenizer...")
    try:
        # Note: First we try loading with use_fast=False because for some models conversion takes too long
        tokenizer = AutoTokenizer.from_pretrained(model_source, trust_remote_code=True, use_fast=False)
    except ValueError:
        tokenizer = AutoTokenizer.from_pretrained(
            model_source,
            trust_remote_code=True,
        )
    special_tokens_dict = {}
    if tokenizer.pad_token is None:
        logger.info("Pad token missing, adding a pad token")
        special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
    if tokenizer.eos_token is None:
        logger.info("EOS token missing, adding a EOS token")
        special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
    if tokenizer.bos_token is None:
        logger.info("BOS token missing, adding a BOS token")
        special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
    if tokenizer.unk_token is None:
        logger.info("UNK token missing, adding a UNK token")
        special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN
    # TODO (chiragjn): Consider adding fake tokens to vocab to pad to multiple of 64. Can provide better throughput
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    return tokenizer, num_new_tokens


def get_max_length(max_length, tokenizer, model_config):
    logger.info("Resolving max_length for truncation...")
    if max_length is None:
        if tokenizer.model_max_length > int(1e6):
            logger.info(f"tokenizer config does not have proper model_max_length set. Looking at model config")
            for length_setting in [
                "max_sequence_length",
                "n_positions",
                "max_position_embeddings",
            ]:
                max_length = getattr(model_config, length_setting, None)
                if max_length:
                    logger.info(f"Assuming value of {length_setting} from model config as max length: {max_length}")
                    break
            if not max_length:
                logger.info(f"Found no max length setting, falling back to default of 512")
                max_length = 512
        else:
            max_length = tokenizer.model_max_length
    logger.info(f"Finally using max_length: {max_length}")
    return max_length


def train(
    *,
    training_arguments: HFTrainingArguments,
    other_arguments: OtherArguments,
    run: Optional[mlfoundry.MlFoundryRun] = None,
):
    if other_arguments.use_lora:
        other_arguments.lora_config = LoraConfig(**json.loads(other_arguments.lora_config))

    set_seed(training_arguments.seed)

    if training_arguments.world_size > 1 and training_arguments.local_rank > 0:
        logger.info("Waiting for main process to load data, process it and fetch any checkpoints ...")
        torch.distributed.barrier()

    train_data, eval_data = get_data(training_arguments=training_arguments, other_arguments=other_arguments)

    last_checkpoint_dir = get_checkpoint_for_resume_if_any(
        training_arguments=training_arguments,
        run=run,
        checkpoint_artifact_name=other_arguments.checkpoint_artifact_name,
    )

    if last_checkpoint_dir:
        model_source = last_checkpoint_dir
    else:
        model_source = other_arguments.model_id

    logger.info("Loading config ...")
    model_config = AutoConfig.from_pretrained(model_source)

    tokenizer, num_new_tokens = get_tokenizer(model_source)

    max_length = get_max_length(max_length=other_arguments.max_length, tokenizer=tokenizer, model_config=model_config)

    train_dataset, eval_dataset = build_dataset(
        train_data=train_data,
        eval_data=eval_data,
        tokenizer=tokenizer,
        max_length=max_length,
        training_arguments=training_arguments,
    )

    if training_arguments.world_size > 1 and training_arguments.local_rank <= 0:
        logger.info("Getting other ranks in sync with main process")
        torch.distributed.barrier()

    model = get_model(model_source, training_arguments=training_arguments)
    if num_new_tokens > 0:
        logger.info("Resizing embeddings layer for newly added tokens")
        model.resize_token_embeddings(len(tokenizer))
        # There are some strategies that also assign unk token as pad token
        # We can also assign the average of all embeddings here for new tokens that got added

    if other_arguments.use_lora:
        logger.info("Applying peft config ...")
        other_arguments.lora_config.inference_mode = False
        model = get_peft_model(model, other_arguments.lora_config)
        if training_arguments.bf16:
            model.to(torch.bfloat16)
        elif training_arguments.fp16:
            model.to(torch.float16)
        if training_arguments.gradient_checkpointing:
            model.enable_input_require_grads()

        model.print_trainable_parameters()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    logger.info("Training...")
    # TODO (chiragjn): Add text generation metrics to `compute_metrics`
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=training_arguments,
        data_collator=SequenceDataCollator(tokenizer, multiple_of=8),
        # Tensor cores are used when tensor dims are multiple of 8
        callbacks=[
            Callback(
                run=run,
                checkpoint_artifact_name=other_arguments.checkpoint_artifact_name,
                log_checkpoints_to_mlfoundry=other_arguments.log_checkpoints_to_mlfoundry,
            )
        ],
    )
    trainer.train(resume_from_checkpoint=last_checkpoint_dir)

    if training_arguments.world_size > 1:
        logger.info("Syncing all processes")
        torch.distributed.barrier()

    logger.info("Saving model...")

    if training_arguments.deepspeed and is_deepspeed_zero3_enabled() and EXPORT_ZERO3_CHECKPOINT_TO_FP32:
        # TODO (chiragjn): Disabled for now
        #  Under ZeRO 3, when checkpointing, each rank saves their own part, in zero format
        #  if "stage3_gather_16bit_weights_on_model_save": true,
        #  then an additional pytorch_model.bin is saved as a 16-bit checkpoint
        #  if we want fp32 pytorch_model.bin then we would have to export separately from the checkpoint in zero forma
        trainer.save_model(output_dir=training_arguments.output_dir)
        if training_arguments.local_rank <= 0:
            fp32_weights_path = os.path.join(training_arguments.output_dir, WEIGHTS_NAME)
            convert_zero_checkpoint_to_fp32_state_dict(trainer.state.best_model_checkpoint, fp32_weights_path)
            cleanup_checkpoints(training_arguments=training_arguments)
    else:
        if training_arguments.local_rank <= 0:
            cleanup_checkpoints(training_arguments=training_arguments)
        trainer.save_model(output_dir=training_arguments.output_dir)

    if training_arguments.world_size > 1:
        logger.info("Syncing all processes")
        torch.distributed.barrier()

    if other_arguments.use_lora:
        if training_arguments.local_rank <= 0:
            logger.info("Merging lora adapter into main model")
            model = AutoPeftModelForCausalLM.from_pretrained(
                training_arguments.output_dir,
                low_cpu_mem_usage=True,
                torch_dtype=get_torch_dtype(training_arguments),
            )
            model = model.merge_and_unload()
            model.save_pretrained(training_arguments.output_dir, safe_serialization=True)
            for filename in ["adapter_config.json", "adapter_model.safetensors", "adapter_model.bin"]:
                file_to_delete = os.path.join(training_arguments.output_dir, filename)
                if os.path.exists(file_to_delete):
                    os.remove(file_to_delete)


def main():
    parser = HfArgumentParser(
        (HFTrainingArguments, OtherArguments),
        description="Fine-tune a language model on a text dataset",
    )
    training_arguments, other_arguments = parser.parse_args_into_dataclasses()
    other_arguments.checkpoint_artifact_name = resolve_checkpoint_artifact_name(
        other_arguments.checkpoint_artifact_name
    )
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S")
    *_, model_name = other_arguments.model_id.rsplit("/", 1)
    model_name = "-".join(["finetuned", model_name, timestamp])
    model_name = model_name.replace(".", "-")

    setup(training_arguments=training_arguments)
    logger.info(f"Training Arguments: {training_arguments}")
    logger.info(f"Arguments: {other_arguments}")

    run = None
    if training_arguments.local_rank <= 0 and other_arguments.report_to_mlfoundry:
        mlfoundry_client = mlfoundry.get_client()
        run = mlfoundry_client.create_run(ml_repo=other_arguments.ml_repo, run_name=f"finetune-{timestamp}")

    if training_arguments.local_rank <= 0 and run:
        run.log_params(vars(other_arguments), flatten_params=True)
        run.log_params(filter_trainer_args_for_logging(training_arguments), flatten_params=True)
        # TODO: there are 110 params in training_arguments, we do not need to log all of them.
        # run.log_params(training_arguments.to_sanitized_dict(), flatten_params=True)

    # Disk space management
    if training_arguments.local_rank <= 0:
        if other_arguments.cleanup_output_dir_on_start and os.path.exists(training_arguments.output_dir):
            logger.warning(f"--cleanup_output_dir_on_start was to set to True, wiping {training_arguments.output_dir}")
            shutil.rmtree(training_arguments.output_dir)

    # We make sure any custom tempdir set by setting `TMPDIR` or equivalent env variables exist
    _tempdir = os.getenv("TMPDIR")
    if _tempdir:
        if os.path.exists(_tempdir) and os.path.isfile(_tempdir):
            raise ValueError("Current `TMPDIR` points to a file path, please set it to a directory path")
        else:
            os.makedirs(_tempdir, exist_ok=True)

    # TODO (chiragjn): Enabled faster kernels for scaled dot product
    # with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=True, enable_mem_efficient=True):
    train(run=run, training_arguments=training_arguments, other_arguments=other_arguments)

    if training_arguments.local_rank <= 0 and run:
        log_model_as_pipeline(
            run=run,
            training_arguments=training_arguments,
            model_name=model_name,
            hf_hub_model_id=other_arguments.model_id,
        )
        run.end()


if __name__ == "__main__":
    main()
