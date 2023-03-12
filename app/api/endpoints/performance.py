"""Endpoint para cálculo de Performance."""
from fastapi import APIRouter
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from .models import Record, Prediction
from .model import MODEL

router = APIRouter(prefix="/performance")

@router.post(
  '/',
  summary="Performance",  
  response_model=list[Prediction],
  response_description="Retorna: \
  (a) A volumetria (quantidade de registros) para cada mês presente na lista de registros; \
  (b)  performance do modelo pré-treinado neste conjunto de registros, indicada pelo valor da área sob a curva ROC.",
)
def get_performance(request: list[Record]):
  request = [r.dict() for r in request]
  records = pd.DataFrame.from_records(request)
  records = records.fillna(np.NaN)
  # convert REF_DATE to datetime type
  records['REF_DATE']=pd.to_datetime(records['REF_DATE'])
  # group records by month
  gp = records.groupby(records['REF_DATE'].dt.month)
  # get each month group
  month_groups = dict(list(gp))
  response = []
  for key in month_groups:
    y_true = month_groups[key]["TARGET"]
    y_pred = MODEL.predict_proba(month_groups[key])[:, 1]
    response.append(Prediction(**{ "mes": key, "volumetria": len(month_groups[key]), "performance": roc_auc_score(y_true, y_pred) }))
  return response
