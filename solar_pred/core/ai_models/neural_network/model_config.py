from dataclasses import dataclass

@dataclass
class ModelConfig:
    target_column: str
    feature_columns: list
    test_size: float = 0.1
    random_state: int = 42


_feature_columns = [
    'soft_amount', 'calculated_lower_bid', 'base_amount_magic_number_A', 
    'base_amount', 'base_amount_magic_number', 'Soft_Amount_Squared'
]

_target_column = 'solar_power'

def load_model_config():
    return ModelConfig(
            target_column=_target_column,
            feature_columns=_feature_columns
            )