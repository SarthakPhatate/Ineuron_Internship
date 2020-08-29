FROM python:3.6

COPY . /usr/src/app

WORKDIR /usr/src/app

RUN pip install -r requirements.txt

EXPOSE 5000

CMD python application.py
