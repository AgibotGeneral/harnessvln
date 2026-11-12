"""Task-specific prompts for navigation and object search."""
from . import prompts_vln, prompts_objnav

PROMPT_REGISTRY = {"VLN": prompts_vln, "OBJNAV": prompts_objnav}


def get_prompts(task_type):
    try:
        return PROMPT_REGISTRY[task_type]
    except KeyError:
        raise ValueError(f"Unsupported task type {task_type!r}; expected VLN or OBJNAV") from None
