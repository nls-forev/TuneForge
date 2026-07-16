"""Eval phase B (local): judge the responses from phase A.

Reads ``responses.parquet`` (from generate_responses) and scores the
fine-tuned response blind on a 1-5 scale via DeepSeek, computes ROUGE-L +
BERTScore against the reference, and runs a swapped-order pairwise win-rate of
fine-tuned vs base. Writes the unified ``metrics.json`` + per-row
``judge_scores.csv``. Needs ``DEEPSEEK_API_KEY`` in the environment.

Heavy deps (openai/rouge/bert) are imported inside the methods.
"""

import json
import os

import pandas as pd

from src.constants import DEEPSEEK_BASE_URL, DEEPSEEK_MODEL
from src.entity.artifact_entity import ModelEvaluationArtifact
from src.entity.config_entity import ModelEvaluationConfig
from src.logger import get_logger

logger = get_logger(__name__)

# Blind absolute grader: sees only instruction, reference, candidate — never
# which model produced the answer.
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

# Blind pairwise grader for win-rate: A/B are the two candidates in an order
# the caller randomises per row to cancel position bias. Length/formatting are
# explicitly neutralised so the verdict reflects medical substance, not verbosity
# — LLM judges otherwise over-reward longer, more heavily-formatted answers.
PAIRWISE_SYSTEM = (
    "You are an impartial medical answer grader. Two candidate answers (A and "
    "B) respond to the same instruction. Decide which is medically better: "
    "judge ONLY on factual correctness, clinical safety, and whether it "
    "actually answers the instruction.\n"
    "Do NOT reward an answer for being longer, more verbose, or more heavily "
    "formatted (headings, bold, bullet lists). A concise correct answer is "
    "better than a long one that adds fluff or errors. Ignore style and "
    "formatting entirely; a tie is appropriate when both are medically "
    "equivalent. Use the reference only as a guide to correct content.\n"
    'Reply with JSON only: {"winner": "A" | "B" | "tie", "reason": "<one sentence>"}'
)

PAIRWISE_USER_TEMPLATE = (
    "Instruction:\n{instruction}\n\n"
    "Reference answer:\n{reference}\n\n"
    "Answer A:\n{answer_a}\n\n"
    "Answer B:\n{answer_b}"
)


class Judge:
    def __init__(
        self,
        model_evaluation_config: ModelEvaluationConfig = ModelEvaluationConfig(),
    ):
        self.model_evaluation_config = model_evaluation_config

    def _deepseek_client(self):
        from openai import OpenAI

        return OpenAI(
            api_key=os.environ["DEEPSEEK_API_KEY"], base_url=DEEPSEEK_BASE_URL
        )

    def _chat_json(self, client, system: str, user: str) -> dict | None:
        import time

        for attempt in range(3):
            try:
                resp = client.chat.completions.create(
                    model=DEEPSEEK_MODEL,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    temperature=0.0,
                    max_tokens=150,
                    response_format={"type": "json_object"},
                )
                return json.loads(resp.choices[0].message.content)
            except Exception:
                logger.exception("judge call failed (attempt %d)", attempt)
                time.sleep(2**attempt)
        return None

    def judge_absolute(self, client, df: pd.DataFrame):
        """Blind 1-5 score of the fine-tuned response."""
        scores: list[int | None] = []
        reasons: list[str] = []
        for i in range(len(df)):
            user_msg = JUDGE_USER_TEMPLATE.format(
                instruction=df["instruction"].iloc[i],
                reference=df["reference"].iloc[i],
                response=df["response"].iloc[i],
            )
            parsed = self._chat_json(client, JUDGE_SYSTEM, user_msg)
            score, reason = None, ""
            if parsed is not None:
                try:
                    s = int(parsed["score"])
                    if 1 <= s <= 5:
                        score, reason = s, str(parsed.get("reason", ""))
                except (KeyError, ValueError, TypeError):
                    logger.warning("bad absolute payload at %d: %s", i, parsed)
            scores.append(score)
            reasons.append(reason)
            if (i + 1) % 20 == 0:
                done = [s for s in scores if s is not None]
                logger.info(
                    "judged %d/%d — running mean %.2f",
                    i + 1,
                    len(df),
                    sum(done) / len(done) if done else float("nan"),
                )
        return scores, reasons

    def judge_pairwise(self, client, df: pd.DataFrame):
        """Swapped-order pairwise win-rate: fine-tuned vs base.

        Each row is judged twice with the answers in both orders so position
        bias cancels. Fine-tuned wins the row only if it is preferred on net
        across the two orderings; otherwise tie/loss.
        """
        outcomes: list[str] = []  # "win" | "tie" | "loss" for fine-tuned
        wins = ties = losses = 0

        for i in range(len(df)):
            ft = df["response"].iloc[i]
            base = df["base_response"].iloc[i]
            ft_points = 0

            # Order 1: A = fine-tuned, B = base
            p1 = self._chat_json(
                client,
                PAIRWISE_SYSTEM,
                PAIRWISE_USER_TEMPLATE.format(
                    instruction=df["instruction"].iloc[i],
                    reference=df["reference"].iloc[i],
                    answer_a=ft,
                    answer_b=base,
                ),
            )
            w1 = (p1 or {}).get("winner", "tie")
            if w1 == "A":
                ft_points += 1
            elif w1 == "B":
                ft_points -= 1

            # Order 2: A = base, B = fine-tuned
            p2 = self._chat_json(
                client,
                PAIRWISE_SYSTEM,
                PAIRWISE_USER_TEMPLATE.format(
                    instruction=df["instruction"].iloc[i],
                    reference=df["reference"].iloc[i],
                    answer_a=base,
                    answer_b=ft,
                ),
            )
            w2 = (p2 or {}).get("winner", "tie")
            if w2 == "B":
                ft_points += 1
            elif w2 == "A":
                ft_points -= 1

            if ft_points > 0:
                outcome = "win"
                wins += 1
            elif ft_points < 0:
                outcome = "loss"
                losses += 1
            else:
                outcome = "tie"
                ties += 1
            outcomes.append(outcome)

            if (i + 1) % 20 == 0:
                logger.info(
                    "pairwise %d/%d — W/T/L %d/%d/%d",
                    i + 1,
                    len(df),
                    wins,
                    ties,
                    losses,
                )

        n = len(df)
        summary = {
            "wins": wins,
            "ties": ties,
            "losses": losses,
            "win_pct": wins / n if n else 0.0,
            "loss_pct": losses / n if n else 0.0,
            "vs": "base Llama-3.1-8B-Instruct",
            "model": DEEPSEEK_MODEL,
        }
        return outcomes, summary

    def compute_rouge_l(self, df: pd.DataFrame) -> list[float]:
        from rouge_score import rouge_scorer

        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        return [
            scorer.score(df["reference"].iloc[i], df["response"].iloc[i])[
                "rougeL"
            ].fmeasure
            for i in range(len(df))
        ]

    def compute_bertscore(self, df: pd.DataFrame) -> list[float]:
        from bert_score import score as bert_score

        _, _, f1 = bert_score(
            df["response"].tolist(),
            df["reference"].tolist(),
            lang="en",
            rescale_with_baseline=True,
            verbose=True,
        )
        return f1.tolist()

    def _save_metrics(self, metrics: dict):
        try:
            os.makedirs(
                self.model_evaluation_config.model_evaluation_dir, exist_ok=True
            )
            with open(
                self.model_evaluation_config.model_evaluation_metrics_file_name, "w"
            ) as f:
                json.dump(metrics, f, indent=2)
            logger.info(
                f"Saved metrics to path: {self.model_evaluation_config.model_evaluation_metrics_file_name}"
            )
        except IOError as e:
            logger.exception(f"Error occurred while saving the file, error: {e}")
            raise

    def run(self) -> ModelEvaluationArtifact:
        """DeepSeek 1-5 + pairwise win-rate + ROUGE-L + BERTScore -> metrics."""
        if not os.environ.get("DEEPSEEK_API_KEY"):
            raise RuntimeError("DEEPSEEK_API_KEY not set — add it to .env")

        df = pd.read_parquet(
            self.model_evaluation_config.model_evaluation_responses_file_name
        )
        logger.info("Judging %d responses.", len(df))

        client = self._deepseek_client()

        scores, reasons = self.judge_absolute(client, df)
        df["judge_score"] = scores
        df["judge_reason"] = reasons

        logger.info("Running pairwise win-rate (fine-tuned vs base)...")
        outcomes, win_rate = self.judge_pairwise(client, df)
        df["pairwise_outcome"] = outcomes

        logger.info("Computing ROUGE-L...")
        df["rouge_l"] = self.compute_rouge_l(df)

        logger.info("Computing BERTScore (downloads model on first run)...")
        df["bertscore_f1"] = self.compute_bertscore(df)

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
            "win_rate": win_rate,
            "rouge_l_mean": float(df["rouge_l"].mean()),
            "bertscore_f1_mean": float(df["bertscore_f1"].mean()),
        }

        self._save_metrics(metrics)
        df.to_csv(
            self.model_evaluation_config.model_evaluation_scores_file_name, index=False
        )
        logger.info(
            "Wrote %s and %s",
            self.model_evaluation_config.model_evaluation_metrics_file_name,
            self.model_evaluation_config.model_evaluation_scores_file_name,
        )
        print(json.dumps(metrics, indent=2))
        return ModelEvaluationArtifact(metrics=metrics)
