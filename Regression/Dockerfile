FROM python:3.9.6

RUN apt-get update -y
RUN apt-get install -y python-pip python-dev build-essential
RUN pip install --upgrade pip
RUN apt-get install -y ghostscript libgs-dev
RUN apt-get install -y libmagickwand-dev imagemagick --fix-missing
RUN apt-get install -y libpng-dev zlib1g-dev libjpeg-dev

RUN mkdir /app

WORKDIR /app

COPY . .

RUN python -m pip install  --upgrade pip

RUN pip install --no-cache-dir -r requirements.txt

CMD ["python", "Deployment/Server/ServerApp.py"]
