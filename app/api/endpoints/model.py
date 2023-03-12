"""Carregamento do modelo treinado."""
import os
import pickle
import logging
from dotenv import load_dotenv

# Load env variables
load_dotenv(override=True)
logger = logging.getLogger(__name__)


DIR_PATH = os.path.dirname(os.path.realpath(__file__))
# Tive que retreinar o modelo, pois estava recebendo um erro (ver arquivo fix_model_error.ipynb)
MODEL_NAME = os.environ.get('MODEL_NAME', 'fixed_model.pkl')
MODEL_PATH = os.path.join(DIR_PATH, '..', '..', MODEL_NAME)
logger.info(f'Loading model from {MODEL_PATH}')

with open(MODEL_PATH, 'rb') as file:
  MODEL = pickle.load(file)
