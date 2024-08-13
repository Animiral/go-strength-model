# How deep is your Go? Full Installation
# Running this container requires GPU access and TCP port access.
# Example ssuming "deepgo" is the container name:
# docker run --gpus all -p 80:8000 deepgo

FROM nvcr.io/nvidia/pytorch:24.05-py3

# settings
ENV APPDIR /app
ENV KATADIR ${APPDIR}/katago
ENV KATAMODEL_NAME kata1-b18c384nbt-s9131461376-d4087399203.bin.gz
ENV KATAGO ${KATADIR}/cpp/katago
ENV KATAMODEL ${KATADIR}/models/${KATAMODEL_NAME}
ENV KATACONFIG ${KATADIR}/cpp/configs/analysis_example.cfg
ENV STRMODEL ${APPDIR}/nets/search2/model.pth
ENV PYTHONPATH "${APPDIR}/python:${APPDIR}/python/model"
ENV FLASK_ENV production
ENV APPPORT 8000

# package lists
RUN apt update

# build tools
RUN apt install -y build-essential cmake

# libzip
RUN apt install -y libzip-dev
ENV LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH}"

# main repo with webapp
RUN pip install --no-cache-dir torch flask gunicorn
RUN git clone -q https://github.com/Animiral/go-strength-model.git ${APPDIR}
RUN ln -s ${APPDIR}/nets/search2/model_4_13_.pth ${STRMODEL}

# katago
WORKDIR ${APPDIR}
RUN git clone -q -b strength-model https://github.com/Animiral/KataGo.git ${KATADIR}
WORKDIR ${KATADIR}/cpp
RUN cmake . -DUSE_BACKEND=CUDA -DBUILD_DISTRIBUTED=1 -DCMAKE_BUILD_TYPE=Release
RUN make -j 4

# kata model
RUN mkdir -p "$(dirname $KATAMODEL)"
RUN curl -o ${KATAMODEL} https://media.katagotraining.org/uploaded/networks/models/kata1/${KATAMODEL_NAME}

# launch web app (it takes KATAGO path, model etc. from ENV)
WORKDIR ${APPDIR}
EXPOSE ${APPPORT}
CMD gunicorn --bind 0.0.0.0:${APPPORT} python.webapp.webapp:app
# CMD python3 ${APPDIR}/python/webapp/webapp.py  # dev server
