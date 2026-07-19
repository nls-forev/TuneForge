"""Eval phase B (local): DeepSeek 1-5 + pairwise win-rate + ROUGE-L + BERTScore.
Needs DEEPSEEK_API_KEY in the environment."""

import json
import os

import pandas as pd

from src.constants import DEEPSEEK_BASE_URL, DEEPSEEK_MODEL
from src.entity.artifact_entity import ModelEvaluationArtifact
from src.entity.config_entity import ModelEvaluationConfig
from src.entity.experiment_config import ExperimentConfig
from src.logger import get_logger

logger = get_logger(__name__)

# Blind absolute grader: never sees which model produced the answer.
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

# Blind pairwise grader, judged in both orders to cancel position bias;
# length/formatting neutralised so the verdict reflects medical substance.
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
        experiment_config: ExperimentConfig | None = None,
    ):
        self.model_evaluation_config = model_evaluation_config
        self.experiment_config = experiment_config or ExperimentConfig.load()

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
                parsed = json.loads(resp.choices[0].message.content)
                if not isinstance(parsed, dict):
                    raise ValueError("judge response must be a JSON object")
                return parsed
            except Exception:
                logger.exception("judge call failed (attempt %d)", attempt)
                time.sleep(2**attempt)
        return None

    @staticmethod
    def valid_absolute_payload(payload: object) -> bool:
        if not isinstance(payload, dict):
            return False
        try:
            return 1 <= int(payload["score"]) <= 5
        except (KeyError, TypeError, ValueError):
            return False

    @staticmethod
    def valid_pairwise_payload(payload: object) -> bool:
        return isinstance(payload, dict) and payload.get("winner") in {"A", "B", "tie"}

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
            if self.valid_absolute_payload(parsed):
                try:
                    s = int(parsed["score"])
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
        """Pairwise win-rate fine-tuned vs base, judged in both orders."""
        outcomes: list[str] = []  # "win" | "tie" | "loss" | "error"
        wins = ties = losses = errors = 0

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
            w1 = (p1 or {}).get("winner")
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
            w2 = (p2 or {}).get("winner")

            # A missing/malformed call is not evidence that the answers tied.
            # Require both order-swapped verdicts before scoring the comparison.
            if not self.valid_pairwise_payload(p1) or not self.valid_pairwise_payload(
                p2
            ):
                logger.warning(
                    "invalid pairwise payload at %d: order1=%s order2=%s",
                    i,
                    p1,
                    p2,
                )
                outcomes.append("error")
                errors += 1
                continue

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
        judged = wins + ties + losses
        summary = {
            "total": n,
            "wins": wins,
            "ties": ties,
            "losses": losses,
            "errors": errors,
            "judged": judged,
            "win_pct": wins / judged if judged else 0.0,
            "loss_pct": losses / judged if judged else 0.0,
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
            resolved_config = self.experiment_config
            responses_path = os.path.join(
                self.model_evaluation_config.model_evaluation_dir,
                "responses.experiment.json",
            )
            if os.path.exists(responses_path):
                resolved_config = ExperimentConfig.load(responses_path)
            resolved_config.save_json(
                os.path.join(
                    self.model_evaluation_config.model_evaluation_dir,
                    "metrics.experiment.json",
                )
            )
            logger.info(
                "Saved metrics to path: %s",
                self.model_evaluation_config.model_evaluation_metrics_file_name,
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
            "n_unique_examples": int(df["source_row_id"].nunique()),
            "generation_seeds": sorted(df["generation_seed"].unique().tolist()),
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
