FROM python:3.9

ENV USER=docker
ENV HOME /home/$USER

RUN mkdir -p $HOME/app
WORKDIR $HOME/app
COPY app/ $HOME/app/
COPY requirements.txt $HOME/
ENV PATH=$HOME/.local/bin:$PATH

RUN /usr/local/bin/python -m pip install --disable-pip-version-check --upgrade pip
RUN pip install -r ../requirements.txt --disable-pip-version-check
EXPOSE 5001

CMD ["uvicorn", "run:app", "--host", "0.0.0.0", "--port", "5001"]