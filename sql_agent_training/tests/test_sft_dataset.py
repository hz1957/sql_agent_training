from sql_agent_training.agent.tokenization import WhitespaceTokenizer
from sql_agent_training.train.sft_dataset import (
    IGNORE_INDEX,
    SftDataCollator,
    SftRecord,
    tokenize_sft_record,
)


def test_tokenize_sft_record_masks_prompt_labels() -> None:
    tokenizer = WhitespaceTokenizer()
    record = SftRecord(uid="1", db_id="music", prompt="schema question SQL:", completion="SELECT name")

    tokenized = tokenize_sft_record(record, tokenizer, max_prompt_length=10, max_response_length=10)

    assert tokenized.input_ids == [1, 2, 3, 4, 5]
    assert tokenized.labels[:3] == [IGNORE_INDEX, IGNORE_INDEX, IGNORE_INDEX]
    assert tokenized.labels[3:] == [4, 5]
    assert tokenized.attention_mask == [1, 1, 1, 1, 1]


def test_tokenize_sft_record_truncates_prompt_from_left_and_completion_from_right() -> None:
    tokenizer = WhitespaceTokenizer()
    record = SftRecord(uid="1", db_id="music", prompt="a b c d", completion="e f g")

    tokenized = tokenize_sft_record(record, tokenizer, max_prompt_length=2, max_response_length=2)

    assert tokenized.input_ids == [3, 4, 5, 6]
    assert tokenized.labels == [IGNORE_INDEX, IGNORE_INDEX, 5, 6]


def test_sft_collator_pads_inputs_and_labels() -> None:
    collator = SftDataCollator(pad_token_id=0)
    batch = collator(
        [
            {"input_ids": [1, 2], "attention_mask": [1, 1], "labels": [-100, 2]},
            {"input_ids": [3], "attention_mask": [1], "labels": [3]},
        ]
    )

    input_ids = batch["input_ids"].tolist() if hasattr(batch["input_ids"], "tolist") else batch["input_ids"]
    labels = batch["labels"].tolist() if hasattr(batch["labels"], "tolist") else batch["labels"]
    attention_mask = (
        batch["attention_mask"].tolist() if hasattr(batch["attention_mask"], "tolist") else batch["attention_mask"]
    )
    assert input_ids == [[1, 2], [3, 0]]
    assert labels == [[-100, 2], [3, -100]]
    assert attention_mask == [[1, 1], [1, 0]]
