import os
import sys
import tqdm
import argparse
import torch
import mlfoundry
import pandas as pd
import evaluate
import logging
from datetime import datetime, timezone
from sentence_transformers import SentenceTransformer
from fuzzywuzzy import fuzz
import numpy as np
import scipy
from typing import List, Union, Any, Dict, Optional

from utils import cached_download

logger = logging.getLogger(__name__)


from vllm import LLM, SamplingParams

WORKING_DIRECTORY = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        ".workdir",
    )
)
ARTIFACTS_DIRECTORY = os.path.join(WORKING_DIRECTORY, "artifacts")
OUTPUTS_DIRECTORY = os.path.join(WORKING_DIRECTORY, "outputs")

os.makedirs(ARTIFACTS_DIRECTORY, exist_ok=True)
os.makedirs(OUTPUTS_DIRECTORY, exist_ok=True)

BATCH_SIZE = 32

@cached_download(ARTIFACTS_DIRECTORY)
def download_model(model_version_fqn, download_dir=None):
    logger.info("Downloading model from mlfoundry")
    client = mlfoundry.get_client()
    model_version = client.get_model_version_by_fqn(fqn=model_version_fqn)
    download_info  = model_version.download(path=download_dir, overwrite=True)
    return download_info.model_dir


@cached_download(ARTIFACTS_DIRECTORY)
def download_data(artifact_version_fqn, download_dir=None):
    logger.info("Downloading artifact from mlfoundry")
    client = mlfoundry.get_client()
    artifact_version = client.get_artifact_version_by_fqn(artifact_version_fqn)
    download_path = artifact_version.download(download_dir, overwrite=True)
    return download_path


def iterate_data_files(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            filepath = os.path.join(root, file)
            filename = os.path.basename(filepath)
            if filename.endswith(".jsonl") and not filename.startswith("."):
                yield filepath


def generate_all_completions(
    dfs,
    model_id, 
    dtype="auto",
    temperature=0.00001,
    top_p=0.95,
    top_k=-1,
    max_tokens=200,
    ignore_eos=False,
    stop=None,
):
    new_dfs = []
    for df in dfs:
        stop = stop or []
        sampling_params = SamplingParams(
            temperature=temperature, 
            top_p=top_p,
            top_k=top_k,
            max_tokens=max_tokens,
            ignore_eos=ignore_eos,
            stop=stop
        )
        llm = LLM(
            model=model_id, 
            tensor_parallel_size=torch.cuda.device_count(),
            dtype=dtype,
            seed=0,
            gpu_memory_utilization=0.95
        )
        prompts = df["prompt"].tolist()
        generated_texts = []
        outputs = llm.generate(prompts, sampling_params)
        for output in tqdm.tqdm(outputs):
            prompt = output.prompt
            generated_text = output.outputs[0].text
            generated_texts.append(generated_text)
        df["generated_text"] = generated_texts
        new_dfs.append(df)
    return new_dfs

_st_model = None

def get_sentence_transformer():
    global _st_model
    if _st_model is None:
        _st_model = SentenceTransformer("BAAI/bge-small-en-v1.5", device="cuda")
    return _st_model


def exact_match_score(df):
    logger.info("Computing exact match")
    exact_match = evaluate.load("exact_match")
    exact_match_scores = []
    for row in tqdm.tqdm(df.itertuples()):
        result = exact_match.compute(
            predictions=[row.generated_text.strip()], 
            references=[row.completion.strip()]
        )
        exact_match_scores.append(result["exact_match"])
    df["exact_match"] = exact_match_scores
    return df

def google_bleu_score(df):
    logger.info("Computing Google Bleu")
    google_bleu = evaluate.load("google_bleu")
    google_bleu_scores = []
    for row in tqdm.tqdm(df.itertuples()):
        result = google_bleu.compute(
            predictions=[row.generated_text.strip()], 
            references=[[row.completion.strip()]]
        )
        google_bleu_scores.append(result["google_bleu"])
    df["google_bleu"] = google_bleu_scores
    return df

def fuzz_ratio(df):
    logger.info("Computing fuzzy ratio")
    fuzz_ratio_scores = []
    for row in tqdm.tqdm(df.itertuples()):
        result = fuzz.ratio(
            row.generated_text.strip(), 
            row.completion.strip()
        )
        fuzz_ratio_scores.append(result)
    df["fuzz_ratio"] = fuzz_ratio_scores
    return df

def meteor_score(df):
    logger.info("Computing Meteor")
    meteor = evaluate.load("meteor")
    meteor_score = evaluate.load("google_bleu")
    meteor_score_scores = []
    for row in tqdm.tqdm(df.itertuples()):
        result = meteor_score.compute(
            predictions=[row.generated_text.strip()], 
            references=[[row.completion.strip()]]
        )
        meteor_score_scores.append(result["meteor"])
    df["meteor_score"] = meteor_score_scores
    return df

def rouge_score(df):
    logger.info("Computing Rouge")
    meteor = evaluate.load("meteor")
    meteor_score_scores = []
    for row in tqdm.tqdm(df.itertuples()):
        result = meteor.compute(
            predictions=[row.generated_text.strip()], 
            references=[[row.completion.strip()]]
        )
        meteor_score_scores.append(result["meteor"])
    df["meteor_score"] = meteor_score_scores
    return df

def bert_score(df):
    logger.info("Computing Bert Score")
    bertscore = evaluate.load("bertscore")
    predictions = []
    references = []
    for row in tqdm.tqdm(df.itertuples()):
        predictions.append(row.generated_text.strip())
        references.append(row.completion.strip())

    result = bertscore.compute(
        predictions=predictions, 
        references=references,
        batch_size=BATCH_SIZE,
         model_type="distilbert-base-uncased"
    )
    df["bert_score_precision"] = result["precision"]
    df["bert_score_recall"] = result["recall"]
    df["bert_score_f1"] = result["f1"]
    
    return df

def cosine_similarity_kernel(
    matrix_a: Union[np.array, scipy.sparse.csr_matrix],
    matrix_b: Union[np.array, scipy.sparse.csr_matrix],
    eps: float = 1e-9
) -> np.array:
    if not (len(matrix_a.shape) == 2 and matrix_a.shape[0] > 0 and matrix_a.shape == matrix_b.shape):
        raise ValueError('`matrix_a` and `matrix_b` should be 2D shaped, non empty and equal in shape')
    if scipy.sparse.issparse(matrix_a) or scipy.sparse.issparse(matrix_b):
        a_norm = (scipy.sparse.linalg.norm(matrix_a, axis=1) + eps).flatten()
        b_norm = (scipy.sparse.linalg.norm(matrix_b, axis=1) + eps).flatten()
        return np.asarray(matrix_a.multiply(matrix_b).sum(axis=1).flatten() / a_norm / b_norm).squeeze(axis=0)
    else:
        a_norm = np.linalg.norm(matrix_a, axis=1) + eps
        b_norm = np.linalg.norm(matrix_b, axis=1) + eps
        return np.sum(matrix_a * matrix_b, axis=1) / a_norm / b_norm

def semantic_similarity(df):
    logger.info("Computing Similarity Score")
    predictions = []
    references = []
    scores = []
    model = get_sentence_transformer()
    for row in tqdm.tqdm(df.itertuples()):
        predictions.append(row.generated_text.strip())
        references.append(row.completion.strip())
    for i in range(0, len(predictions), BATCH_SIZE):
        pchunk = predictions[i:i + BATCH_SIZE]
        rchunk = references[i:i + BATCH_SIZE]
        embeddings_1 = model.encode(pchunk, normalize_embeddings=True)
        embeddings_2 = model.encode(rchunk, normalize_embeddings=True)
        similarity = cosine_similarity_kernel(embeddings_1, embeddings_2).tolist()
        scores.extend(similarity)
    df["similarity"] = scores
    return df


def add_eval_scores(df):
    df = exact_match_score(df)
    df = google_bleu_score(df)
    df = fuzz_ratio(df)
    df = bert_score(df)
    df = semantic_similarity(df)
    return df



def main():
    def _boolify(value):
        return value in ("true", "True", "yes", "1")

    def _csv(value):
        if value:
            return value.split(",")
        return value
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--ml_repo", type=str, required=True)
    parser.add_argument("--run_name", type=str, default="auto")
    parser.add_argument("--model_id", type=str, required=True)
    parser.add_argument("--data_uri", type=str, required=True)
    parser.add_argument("--model_dtype", type=str, default="auto")
    parser.add_argument("--temperature", type=float, default=0.00001)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=-1)
    parser.add_argument("--max_tokens", type=int, default=200)
    parser.add_argument("--ignore_eos", type=_boolify, default="false")
    parser.add_argument("--stop", type=_csv, default=None)

    args = parser.parse_args()
    logger.setLevel(logging.DEBUG)
    handler = logging.StreamHandler(stream=sys.stdout)
    formatter = logging.Formatter(fmt="%(asctime)s %(levelname)s - %(module)s:%(funcName)s:%(lineno)d - %(message)s")
    handler.setFormatter(formatter)
    handler.setLevel(logging.INFO)
    logger.addHandler(handler)

    if args.run_name == "auto":
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S")
        args.run_name = f"eval-{timestamp}"

    client = mlfoundry.get_client()
    run = client.create_run(ml_repo=args.ml_repo, run_name=args.run_name)

    run.log_params(param_dict=dict(
        data_uri=args.data_uri,
        model_id=args.model_id, 
        dtype=args.model_dtype,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_tokens=args.max_tokens,
        ignore_eos=args.ignore_eos,
        stop=args.stop
    ))
    
    if args.model_id.startswith("model:"):
        args.model_id = download_model(args.model_id,)

    if args.data_uri.startswith("artifact:"):
        args.data_uri = download_data(args.data_uri)

    filepaths = []
    dfs = []
    for filepath in iterate_data_files(args.data_uri):
        filename = os.path.splitext(os.path.basename(filepath))[0]
        logger.info(f"Loading file {filepath} ...")
        df = pd.read_json(filepath, lines=True)
        df.dropna(inplace=True)
        df = df[["prompt", "completion"]]
        filepaths.append(filepath)
        dfs.append(df)

    dfs = generate_all_completions(
        dfs,
        model_id=args.model_id, 
        dtype=args.model_dtype,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_tokens=args.max_tokens,
        ignore_eos=args.ignore_eos,
        stop=args.stop,
    )
    for filepath, df in zip(filepaths, dfs):
        add_eval_scores(df)
        output_filename = os.path.splitext(os.path.basename(filename))[0] + ".output.csv"
        output_filepath = os.path.join(OUTPUTS_DIRECTORY, output_filename) 
        df.to_csv(output_filepath, index=False)

    run.log_artifact(
        name="eval-results", 
        artifact_paths=[(OUTPUTS_DIRECTORY, None)]
    )
    run.end()
  

if __name__ == "__main__":
    main()

