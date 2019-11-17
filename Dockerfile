FROM python:3
COPY . /bhacks2019
WORKDIR /bhacks2019
RUN pip3 install -r ./requirements.txt
CMD ["python3", "app.py"]

