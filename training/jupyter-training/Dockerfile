# JupyterLab image with YOLO training notebook at ~/work/yolo_training.ipynb
# Build context is training/; notebook source in Git is training/yolo_training.ipynb.
FROM quay.io/jupyter/scipy-notebook:python-3.11

USER root
RUN mkdir -p /opt/bundled-training /home/jovyan/work
COPY yolo_training.ipynb /opt/bundled-training/yolo_training.ipynb
COPY yolo_training.ipynb /home/jovyan/work/yolo_training.ipynb
RUN chmod -R a+rX /opt/bundled-training
RUN chown -R "${NB_UID}:${NB_GID}" /home/jovyan/work
# jupyterlab-git is enabled under /opt/conda/etc/jupyter; Helm-only disable in /etc/jupyter does not reliably override merged config, so Lab still shows errors. Re-enable Git later: remove this RUN, rebuild, and set jupyter-training.jupyterGit.enabled=true (install git on PATH as needed).
RUN /opt/conda/bin/pip uninstall -y jupyterlab-git || true
RUN /opt/conda/bin/pip install --no-cache-dir ultralytics
# docker-stacks often leaves /home/jovyan mode 700; OpenShift arbitrary UID cannot traverse it, so pathlib.resolve() on terminal cwd raises PermissionError.
RUN chmod 755 /home/jovyan

USER ${NB_UID}
