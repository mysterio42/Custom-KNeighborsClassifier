FROM python:3.6.4-slim
WORKDIR /usr/src/app
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["python","app.py"]
