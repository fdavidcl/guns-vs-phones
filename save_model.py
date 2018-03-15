import logging

from keras.models import model_from_json

logger = logging.getLogger('save_model')

def save_my_model(model):
    logger.info("Saving model...")
    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
    model.save_weights("model.h5")

def load_my_model(model = 'model.json'):
    logger.info('Loading model and weights...')
    json_file = open(model, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights('model.h5')

    return loaded_model
