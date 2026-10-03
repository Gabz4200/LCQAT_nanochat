"""
The ARC dataset from Allen AI.
https://huggingface.co/datasets/allenai/ai2_arc
"""

from nanochat.tasks.common import (
    EVAL_TYPE_CATEGORICAL,
    Task,
    categorical_match,
    load_hub_dataset,
    render_mc,
)


class ARC(Task):
    def __init__(self, subset, split, **kwargs):
        super().__init__(**kwargs)
        assert subset in ["ARC-Easy", "ARC-Challenge"], (
            "ARC subset must be ARC-Easy or ARC-Challenge"
        )
        assert split in ["train", "validation", "test"], (
            "ARC split must be train|validation|test"
        )
        self.ds = load_hub_dataset("allenai/ai2_arc", subset, split=split).shuffle(
            seed=42
        )

    @property
    def eval_type(self):
        return EVAL_TYPE_CATEGORICAL

    def num_examples(self):
        return len(self.ds)

    def get_example(self, index):
        row = self.ds[index]
        question = row["question"]  # the question text
        choices = row["choices"]["text"]  # the text of each choice
        answer_string = row["answerKey"]  # e.g. "A", "B", "C", "D"
        letters = row["choices"]["label"]  # e.g. ["A", "B", "C", "D"]
        assert answer_string in letters, (
            f"ARC answer {answer_string} must be one of {letters}"
        )  # sanity check
        user_message = render_mc(question, letters, choices)
        messages = [
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": answer_string},
        ]
        conversation = {
            "messages": messages,
            "letters": letters,  # useful during evaluation, so we can narrow and clamp the assistant prediction to one of the letters
        }
        return conversation

    def evaluate(self, conversation, assistant_response):
        return categorical_match(
            conversation, assistant_response, conversation["letters"], "ARC"
        )
