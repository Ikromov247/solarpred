from sklearn.preprocessing import StandardScaler



MODELS_CONFIG = {

        "GENERAL_CONFIG": {
            "val_size": 44,
            "separate_val_set": False,
            "target_col": "solar_power",
            "features_to_use": ['global_tilted_irradiance_instant', 
                                'global_tilted_irradiance', 
                                'cloud_cover_mid', 
                                'cloud_cover_high',
                                'uv_index',
                                'diffuse_radiation',
                                'direct_radiation_instant'
                                ], # if you want to use all features, set this to ['all']
            "scaler": StandardScaler(),
            "normalize": True,
            "deterministic": True,
            "output_boundaries": {}
        },

        "neural_network": {
            "n_epochs": 14,
            "batch_size": 32,
            "learning_rate": 0.002,
            "dropout_rate": 0.1,
            "device": "cuda",
        }

}

def get_model_config(model_name: str,  max_cap: float = 4.0, min_cap: float = 40.0):
    # Add general configurations to all model configurations
    for key, _ in MODELS_CONFIG.items():
        MODELS_CONFIG[key].update(MODELS_CONFIG["GENERAL_CONFIG"])
    models_config = MODELS_CONFIG[model_name]

    # Add output boundaries to the model configuration
    models_config["output_boundaries"]["upper"] = max_cap
    models_config["output_boundaries"]["lower"] = min_cap

    return models_config