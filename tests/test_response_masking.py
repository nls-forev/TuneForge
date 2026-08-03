from src.components.model_trainer import INSTRUCTION_MASK, RESPONSE_MASK


def test_response_masking_matches_llama_chat_headers():
    assert INSTRUCTION_MASK == "<|start_header_id|>user<|end_header_id|>\n\n"
    assert RESPONSE_MASK == "<|start_header_id|>assistant<|end_header_id|>\n\n"
