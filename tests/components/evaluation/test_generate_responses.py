import pandas as pd

from src.components.evaluation.generate_responses import GenerateResponses


def test_build_prompts_combines_instruction_and_real_input():
    df = pd.DataFrame(
        {
            "instruction": ["Explain this result. "],
            "input": [" CBC is normal. "],
        }
    )

    result = GenerateResponses._build_prompts(df)

    assert result["user_prompt"].tolist() == ["Explain this result.\n\nCBC is normal."]


def test_build_prompts_omits_noinput_sentinel_without_mutating_source():
    df = pd.DataFrame(
        {
            "instruction": ["Describe hypertension. "],
            "input": ['  "<No Input>"  '],
        }
    )

    result = GenerateResponses._build_prompts(df)

    assert result["user_prompt"].tolist() == ["Describe hypertension."]
    assert "user_prompt" not in df.columns
