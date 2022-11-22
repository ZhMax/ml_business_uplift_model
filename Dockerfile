FROM python:3.7
LABEL maintainer="zhelninmax@gmail.com"
COPY . /app_docker
WORKDIR /app_docker
RUN pip install -r requirements.txt

ENTRYPOINT ["python"]
CMD ["/app_docker/app/flask_server.py"]