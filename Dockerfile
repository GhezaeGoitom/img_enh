FROM python:3.9

RUN mkdir /app
COPY . /app
WORKDIR /app

RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y


RUN pip install -r dev_requirements.txt

EXPOSE 8000
WORKDIR /app/src/service
CMD ["uvicorn","main:app","--host","0.0.0.0","--port","8000"]


# build : docker build -t image-enhancement:0.1 .
# run : docker run -p 8000:8000 --name my_app image-enhancement:0.1