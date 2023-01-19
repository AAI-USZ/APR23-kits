FROM python:3.8.6

WORKDIR /app/aprkits

COPY ./requirements.txt /app/aprkits/requirements.txt
RUN pip install -r ./requirements.txt
