import verifiers as vf
from verifiers.utils.data_utils import BOXED_SYSTEM_PROMPT, load_example_dataset


def load_environment(
    system_prompt: str = BOXED_SYSTEM_PROMPT,
    eval_start: int = 0,
    num_eval_examples: int = -1,
    num_train_examples: int = -1,
):
    if eval_start < 0:
        raise ValueError("eval_start must be non-negative")

    def build_dataset():
        dataset = load_example_dataset("gsm8k", split="train")
        if num_train_examples != -1:
            dataset = dataset.select(range(num_train_examples))
        return dataset

    def build_eval_dataset():
        dataset = load_example_dataset("gsm8k", split="test")
        eval_stop = len(dataset)
        if num_eval_examples != -1:
            eval_stop = eval_start + num_eval_examples
        if eval_stop > len(dataset):
            raise ValueError(
                f"requested eval slice [{eval_start}, {eval_stop}) exceeds "
                f"dataset length {len(dataset)}"
            )
        dataset = dataset.select(range(eval_start, eval_stop))
        return dataset.add_column("example_id", range(eval_start, eval_stop))

    rubric = vf.MathRubric()
    return vf.SingleTurnEnv(
        dataset=build_dataset,
        eval_dataset=build_eval_dataset,
        system_prompt=system_prompt,
        parser=rubric.parser,
        rubric=rubric,
    )
