FROM conda/miniconda3
RUN mkdir app
COPY ./app /app
COPY ./requirements.txt /requirements.txt
RUN  conda install pytorch torchvision -c pytorch
RUN pip install -r requirements.txt
RUN apt update && apt install -y libsm6 libxext6 && apt-get install -y libxrender-dev && apt-get install -y libgtk2.0-dev
RUN apt-get install ffmpeg -y
WORKDIR /app
EXPOSE 8888