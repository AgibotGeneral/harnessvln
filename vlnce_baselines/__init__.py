"""HarnessVLN public package; simulator registration is explicit.

Importing the harness, prompts or API utilities does not initialize Habitat.
Evaluation entry points must call :func:`register_extensions` before looking
up a trainer, environment or policy in Habitat's registries.
"""

__version__ = "3.0.0"


def register_extensions():
    """Register simulation components once per process (imports are cached)."""
    from importlib import import_module

    for module in (
        "vlnce_baselines.ZS_Evaluator_mp",
        "vlnce_baselines.env.environments",
        "vlnce_baselines.models.Policy",
    ):
        import_module(module)
