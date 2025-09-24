
def load_nr_model(weights_dir):
    from .model import NeuralNetwork
    try:
        return NeuralNetwork.load_from_file(file_directory=weights_dir)
    except FileNotFoundError:
        # load a new model
        from solar_pred.core.ai_models._models_config import get_model_config
        model_config = get_model_config(model_name="neural_network")
        return NeuralNetwork(model_CONFIG=model_config)
