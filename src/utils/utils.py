import pickle as pk
from settings import out_dir


def save_model(model, model_name):
    """saving trained model in out_dir folder (see settings.py) with model_name"""
    with open(out_dir + model_name + ".pk", "wb") as file:
        pk.dump(model, file)

    return True


def save_predictions(predictions, model_name):
	"""saving trained model in out_dir folder (see settings.py) with model_name"""
	with open(out_dir + model_name + "_predictions.pk", "wb") as file:
        pk.dump(predictions, file)

    return True


def load_model(model_name):
    """loading trained model in out_dir folder (see settings.py) with model_name"""
    with open(out_dir + model_name + ".pk", "rb") as file:
        model = pk.load(file)

    return model
