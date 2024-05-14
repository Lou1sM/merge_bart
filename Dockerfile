FROM python:latest
#FROM continuumio/miniconda3
#FROM mambaorg/micromamba
#FROM mambaorg/micromamba:1.5.8
WORKDIR /tmp
COPY train.py ./
COPY utils.py ./
COPY manual_requirements.sh ./
COPY environment.yml ./
COPY models/ ./models
COPY datasets/ ./datasets

#COPY test.py ./

#RUN micromamba install -y -n base -f environment.yml && \
    #micromamba clean --all --yes

#ENV PATH="/home/louis/miniforge3/condabin:${PATH}"
#RUN echo ${PATH}
#RUN [ "micromamba", "env", "create", "-f", "environment.yml" ]
#SHELL ["micromamba", "run", "-n", "merge-bart", "/bin/bash", "-c"]
RUN bash manual_requirements.sh
ENTRYPOINT ["python", "train.py", "--n-train=0"]
#ENTRYPOINT ["python", "test.py"]
#ENTRYPOINT ["ls", "datasets"]
