import logging

from keras.models import model_from_json

logger = logging.getLogger('save_model')
logger.setLevel(logging.INFO)

def save_my_model(model,
                  modelname = 'model.json',
                  w = 'model.h5',
                  path = 'models/'):
    filename = path + modelname + '.json'

    model_json = model.to_json()
    with open(filename, "w") as json_file:
        logger.info('Writing %s...', filename)
        json_file.write(model_json)
    logger.info('Writing %s...', path + w)
    model.save_weights(path + w + '.h5')

def load_my_model(model = 'model.json',
                  weights = 'model.h5',
                  path = 'models/'):
    model = path + model
    model_w = path + weights

    logger.info('reading %s...', model)
    json_file = open(model, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    logger.info('reading %s...', model_w)
    loaded_model.load_weights(model_w)

    return loaded_model
