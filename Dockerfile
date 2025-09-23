FROM python:3.10

# all files will be copied here 
WORKDIR /app

COPY pipeline_skeleton/ /app/pipeline_skeleton
COPY requirements.txt .

RUN pip3 install --no-cache-dir -r requirements.txt

ENTRYPOINT [ "python3", "./pipeline_skeleton/main.py"]