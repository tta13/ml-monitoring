"""Endpoint para cálculo de aderência."""
from fastapi import APIRouter, HTTPException
import os
import gzip
import pandas as pd
from .models import Path, Adherence
from scipy.stats import ks_2samp
from .model import MODEL
import numpy as np

DIR_PATH = os.path.dirname(os.path.realpath(__file__))
TEST_DATA_PATH = os.path.join(DIR_PATH, '..', '..', 'datasets', 'credit_01', 'test.gz')
with gzip.open(TEST_DATA_PATH, 'r') as file:
  test_data = pd.read_csv(file)

test_data['SCORE'] = MODEL.predict_proba(test_data)[:, 1]

router = APIRouter(prefix="/aderencia")

@router.post(
  '/',
  summary="Aderência",
  response_model=Adherence,
  response_description="Indica o quanto a distribuição de scores em uma certa base está distante, \
    ou diferente, da distribuição vista na base de Teste da modelagem utilizando o teste \
    estatístico de Kolmogorov-Smirnov (KS).",
)
def get_adherence(request: Path):
  path = request.path
  if path.endswith('gz'):
    with gzip.open(path, 'r') as file:
      data = pd.read_csv(file)
  elif path.endswith('csv'):
    data = pd.read_csv(path)
  elif path.endswith('json'):
    data = pd.read_json(path)
  else:
    raise HTTPException(status_code=400, detail="Unsupported file extension")
  
  data = data.fillna(np.NaN)
  data['SCORE'] = MODEL.predict_proba(data)[:, 1]

  result = ks_2samp(test_data['SCORE'], data['SCORE'])
  
  return Adherence(**{
    'statistic': result.statistic,
    'p_value': result.pvalue
  })
