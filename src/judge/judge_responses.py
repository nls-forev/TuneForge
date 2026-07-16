"""Judge run4 responses: DeepSeek blind 1-5 + ROUGE-L + BERTScore.

Phase B of the LLM-as-judge eval: runs locally against
artifact/model_evaluation/responses.parquet from phase A. Needs
DEEPSEEK_API_KEY in the environment (.env). Writes judge_metrics.json and
judge_scores.csv next to the input.

Run: PYTHONPATH=. uv run --group judge python -m src.judge.judge_responses
"""

import json
import os
import sys
import time

from dotenv import load_dotenv

load_dotenv()

import pandas as pd  # noqa: E402

from src.logger import configure_logger, get_logger  # noqa: E402

configure_logger()
logger = get_logger(__name__)

RESPONSES_PATH = "artifact/model_evaluation/responses.parquet"
METRICS_PATH = "artifact/model_evaluation/judge_metrics.json"
SCORES_PATH = "artifact/model_evaluation/judge_scores.csv"

DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_MODEL = "deepseek-chat"

# Blind: the judge sees only the question, a reference answer, and the
# candidate answer — never which model (or that a fine-tune is involved).
JUDGE_SYSTEM = (
    "You are an impartial medical answer grader. Score the candidate answer "
    "to the given instruction on a 1-5 scale:\n"
    "5 = medically accurate, complete, directly answers the instruction\n"
    "4 = accurate with minor omissions or slight verbosity\n"
    "3 = partially correct; notable omissions or minor inaccuracies\n"
    "2 = mostly unhelpful or contains a significant medical error\n"
    "1 = wrong, harmful, off-topic, or incoherent\n"
    "Use the reference answer as a guide to what a good answer contains, but "
    "a candidate may phrase things differently and still score 5.\n"
    'Reply with JSON only: {"score": <1-5>, "reason": "<one sentence>"}'
)

JUDGE_USER_TEMPLATE = (
    "Instruction:\n{instruction}\n\n"
    "Reference answer:\n{reference}\n\n"
    "Candidate answer:\n{response}"
)


def judge_deepseek(df: pd.DataFrame) -> tuple[list[int | None], list[str]]:
    from openai import OpenAI

    client = OpenAI(api_key=os.environ["DEEPSEEK_API_KEY"], base_url=DEEPSEEK_BASE_URL)

    scores: list[int | None] = []
    reasons: list[str] = []
    for i in range(len(df)):
        user_msg = JUDGE_USER_TEMPLATE.format(
            instruction=df["instruction"].iloc[i],
            reference=df["reference"].iloc[i],
            response=df["response"].iloc[i],
        )
        score, reason = None, ""
        for attempt in range(3):
            try:
                resp = client.chat.completions.create(
                    model=DEEPSEEK_MODEL,
                    messages=[
                        {"role": "system", "content": JUDGE_SYSTEM},
                        {"role": "user", "content": user_msg},
                    ],
                    temperature=0.0,
                    max_tokens=150,
                    response_format={"type": "json_object"},
                )
                parsed = json.loads(resp.choices[0].message.content)
                s = int(parsed["score"])
                if 1 <= s <= 5:
                    score, reason = s, str(parsed.get("reason", ""))
                break
            except Exception:
                logger.exception("judge call %d failed (attempt %d)", i, attempt)
                time.sleep(2**attempt)
        scores.append(score)
        reasons.append(reason)
        if (i + 1) % 20 == 0:
            done = [s for s in scores if s is not None]
            logger.info(
                "judged %d/%d — running mean %.2f",
                i + 1,
                len(df),
                sum(done) / len(done),
            )
    return scores, reasons


def compute_rouge_l(df: pd.DataFrame) -> list[float]:
    from rouge_score import rouge_scorer

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    return [
        scorer.score(df["reference"].iloc[i], df["response"].iloc[i])["rougeL"].fmeasure
        for i in range(len(df))
    ]


def compute_bertscore(df: pd.DataFrame) -> list[float]:
    from bert_score import score as bert_score

    _, _, f1 = bert_score(
        df["response"].tolist(),
        df["reference"].tolist(),
        lang="en",
        rescale_with_baseline=True,
        verbose=True,
    )
    return f1.tolist()


def main() -> int:
    if not os.environ.get("DEEPSEEK_API_KEY"):
        logger.error("DEEPSEEK_API_KEY not set — add it to .env")
        return 1

    df = pd.read_parquet(RESPONSES_PATH)
    logger.info("Judging %d responses.", len(df))

    scores, reasons = judge_deepseek(df)
    df["judge_score"] = scores
    df["judge_reason"] = reasons

    logger.info("Computing ROUGE-L...")
    df["rouge_l"] = compute_rouge_l(df)

    logger.info("Computing BERTScore (downloads model on first run)...")
    df["bertscore_f1"] = compute_bertscore(df)

    valid = df[df["judge_score"].notna()]
    metrics = {
        "n": len(df),
        "judge": {
            "mean": float(valid["judge_score"].mean()),
            "median": float(valid["judge_score"].median()),
            "distribution": {
                str(k): int(v)
                for k, v in valid["judge_score"].value_counts().sort_index().items()
            },
            "failed_calls": int(df["judge_score"].isna().sum()),
            "model": DEEPSEEK_MODEL,
        },
        "rouge_l_mean": float(df["rouge_l"].mean()),
        "bertscore_f1_mean": float(df["bertscore_f1"].mean()),
    }

    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)
    df.to_csv(SCORES_PATH, index=False)
    logger.info("Wrote %s and %s", METRICS_PATH, SCORES_PATH)
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
