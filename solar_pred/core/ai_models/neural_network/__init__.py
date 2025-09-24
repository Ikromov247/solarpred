
def load_nr_model(weighs_dir):
    from .model import NeuralNetwork
    try:
        return NeuralNetwork.load(weighs_dir)
    except FileNotFoundError:
        # load a new model
        from core.ai_models._models_config import get_model_config
        model_config = get_model_config(model_name="neural_network")
        return NeuralNetwork(model_CONFIG=model_config)
