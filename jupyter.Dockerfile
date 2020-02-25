FROM jupyter/scipy-notebook

COPY requirements.txt /home/jovyan/work/requirements.txt
RUN conda config --add channels conda-forge && conda config --set channel_priority strict
RUN conda install --file /home/jovyan/work/requirements.txt
