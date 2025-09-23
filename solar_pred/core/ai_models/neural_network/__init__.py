
def load_nr_model(weighs_path):
    from .model import ModelClass
    return ModelClass.load(weighs_path)
