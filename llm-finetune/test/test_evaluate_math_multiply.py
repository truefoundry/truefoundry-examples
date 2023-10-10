import argparse
import json
import logging
import os
import re
import sys
import tempfile
from argparse import ArgumentParser
from dataclasses import dataclass, field
from typing import Optional

import mlfoundry
from cloudfiles import CloudFile
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    TrainingArguments,
    pipeline,
)

logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter(fmt=f"[%(asctime)s]  %(message)s")
handler.setFormatter(formatter)
logger.setLevel(logging.DEBUG)
logger.addHandler(handler)
PROMPT_KEY = "prompt"
COMPLETION_KEY = "completion"


@dataclass
class HFTrainingArguments(TrainingArguments):
    pass


@dataclass
class Args:
    model_id: str = field(metadata={"help": "Huggingface hub model ID"})
    ml_repo: str = field(metadata={"help": "ML Repo to put the model to"})
    eval_data: Optional[str] = field(
        default=None,
        metadata={"help": "URL to the jsonl evaluation dataset. Overrides eval_size. Leave as NA if not available"},
    )
    tfy_run_name: Optional[str] = field(
        default=None, metadata={"help": "Run name for the job to save the metrics and "}
    )
    fine_tune: Optional[bool] = field(default=False, metadata={"help": "Weather the model is finetuned model or not"})


class DataValidationException(Exception):
    pass


def get_run(args: Args) -> mlfoundry.MlFoundryRun:
    mlfoundry_client = mlfoundry.get_client()
    run = mlfoundry_client.get_run_by_name(ml_repo=args.ml_repo, run_name=args.tfy_run_name)
    logger.info(
        f"The evalutaion metric will be logged to run {run.run_name} and dashboard link for run is {run.dashboard_link}"
    )
    return run


def load_pipeline(args: Args) -> pipeline:
    logger.info("loading model ....")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id, device_map="auto", trust_remote_code=False, revision="main"
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, use_fast=True)
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=4028,
        do_sample=True,
        temperature=0.01,
        top_k=1,
        return_full_text=False,
    )

    return pipe


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


def load_data(path):
    data = []
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
        for line_no, line in enumerate(lines, start=1):
            if not line.strip():
                continue
            try:
                datapoint_dict = json.loads(line)
            except json.decoder.JSONDecodeError as je:
                raise DataValidationException(
                    f"Failed to parse json line on line number {line_no}. Line: {line[:150]}..."
                ) from je
            else:
                for key in (PROMPT_KEY, COMPLETION_KEY):
                    if key not in datapoint_dict:
                        raise DataValidationException(
                            f"Required key `{key}` is missing from json line on line number {line_no}. Line: {line[:150]}..."
                        )
                    if not isinstance(datapoint_dict[key], str) or not datapoint_dict[key]:
                        raise DataValidationException(
                            f"Value for `{key}` is not a non-empty string on line on line number {line_no}. Line: {line[:150]}..."
                        )

                datapoint_dict = {
                    PROMPT_KEY: datapoint_dict[PROMPT_KEY],
                    COMPLETION_KEY: datapoint_dict[COMPLETION_KEY],
                }
                data.append(datapoint_dict)
    return data


def get_answer(output):
    match = re.search(r"'answer': (\d+)", output)
    if match:
        return match.group(1)

    return "0.1"


def evaluate(args: Args):
    run = get_run(args=args)
    pipeline = load_pipeline(args=args)
    data = load_data(args.eval_data)
    correct = 0
    total = 0
    logger.info("Starting the evaluation...")
    for row in data:
        output = pipeline(row[PROMPT_KEY])[0]["generated_text"]
        predicted = get_answer(output)
        actual = get_answer(row[COMPLETION_KEY])
        if predicted == actual:
            correct += 1
        total += 1
    accuracy = correct / total
    if args.fine_tune:
        run.log_metrics({"finetune-test-accuracy": accuracy})
        logger.info(f"The accuracy of the Finetuned model is {accuracy}")
    else:
        logger.info(f"The accuracy of the Norma model is {accuracy}")
        run.log_metrics({"normal-test-accuracy": accuracy})

    logger.info(
        f"The evaluation is complete and the metrics have been logged to the run {run.run_name} where dashboard link is {run.dashboard_link}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, required=True)
    parser.add_argument("--ml_repo", type=str, required=True)
    parser.add_argument("--eval_data", type=str, required=True)
    parser.add_argument("--tfy_run_name", type=str, required=True)
    parser.add_argument("--fine_tune", type=str, required=True, default=False)
    args = parser.parse_args()
    args = Args(
        model_id=args.model_id,
        ml_repo=args.ml_repo,
        eval_data=args.eval_data,
        tfy_run_name=args.tfy_run_name,
        fine_tune=args.fine_tune,
    )
    evaluate(args)
