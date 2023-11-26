from training_utils import get_trained_model
from my_model import ModelFactory
from model_constants import ModelParameters
if __name__ == '__main__':
    input_file = open("", 'r')
    input_string = input_file.read()
    input_file.close()
    model_factory = ModelFactory(ModelParameters(100, 100), ModelFactory.ModelType.SPLIT_READY_MODEL)
    model = get_trained_model(input_string, model_factory, 100, 20, 0)