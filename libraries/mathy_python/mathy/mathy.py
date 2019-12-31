from .agents.policy_value_model import load_policy_value_model, PolicyValueModel
from .agents.base_config import BaseConfig
from typing import Optional


class Mathy:
    """The standard interface for working with Mathy models and agents"""

    config: BaseConfig
    model: PolicyValueModel

    def __init__(
        self,
        *,
        model_path: Optional[str] = None,
        model: Optional[PolicyValueModel] = None
    ):
        if model_path is not None:
            self.model, self.config = load_policy_value_model(model_path)
        elif model is not None:
            if not isinstance(model, PolicyValueModel):
                raise ValueError("model must derive PolicyValueModel for compatibility")
            self.model = model
