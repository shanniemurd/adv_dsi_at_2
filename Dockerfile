FROM jupyter/scipy-notebook:0ce64578df46

## FOR PYTORCH

RUN pip install torch==1.9.0+cpu torchvision==0.10.0+cpu torchtext==0.10.0 -f https://download.pytorch.org/whl/torch_stable.html

## FOR SKLEARN PIPELINE

RUN conda install yellowbrick

RUN pip install mlflow==1.13

RUN pip install psycopg2-binary==2.8.5


## SET ENVIRONMENT

ENV PYTHONPATH "${PYTHONPATH}:/home/jovyan/work"

RUN echo "export PYTHONPATH=/home/jovyan/work" >> ~/.bashrc

WORKDIR /home/jovyan/work