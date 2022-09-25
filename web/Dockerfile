FROM python:3.9.7

WORKDIR /app

COPY requirements.txt .
RUN apt-get update && apt-get install -y python3-opencv
RUN pip install -r requirements.txt

COPY . .
CMD ["python", "app.py"]