# App

Para rodar a aplicação utilizando o docker:
```console
$ docker build -t neurotech/ml-monitoring-app .
$ docker run -d --name app -p 8001:8001 --env MODEL_NAME=xgb_model.pkl neurotech/ml-monitoring-app 
```

A documentação pode ser acessada em: http://localhost:8001/docs
