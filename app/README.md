# App

Para rodar a aplicação utilizando o docker:
```console
$ docker build -t neurotech/ml-monitoring-app .
$ docker run -d --name app -p 8001:8001 ml-monitoring-app 
```
