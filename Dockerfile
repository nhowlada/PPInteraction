FROM nvidia/cuda:11.3.1-base-ubuntu20.04

# Remove any third-party apt sources to avoid issues with expiring keys.
RUN rm -f /etc/apt/sources.list.d/*.list

# Install some basic utilities
RUN apt-get update && apt-get install -y \
    curl \
    wget \
    ca-certificates \
    sudo \
    git \
    bzip2 \
    libx11-6 \
    && rm -rf /var/lib/apt/lists/*

# Create a working directory
RUN mkdir /app
WORKDIR /app

# Create a non-root user and switch to it
RUN adduser --disabled-password --gecos '' --shell /bin/bash user \
 && chown -R user:user /app \
 && echo "user ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/90-user
USER user

# All users can use /home/user as their home directory
ENV HOME=/home/user
RUN chmod 777 /home/user

# Set up the Conda environment
ENV CONDA_AUTO_UPDATE_CONDA=false \
    PATH=/app/miniconda/bin:$PATH

COPY requirements.txt /app/requirements.txt

RUN wget https://repo.continuum.io/miniconda/Miniconda3-3.7.0-Linux-x86_64.sh -O ./miniconda.sh
RUN chmod ouga+xw ./miniconda.sh
RUN bash ./miniconda.sh -b -p ./miniconda

RUN conda update conda
RUN conda create -n PPInteraction python=3.9.7
RUN /bin/bash -c "source activate PPInteraction && pip3 install -r requirements.txt"

COPY . /app
RUN sudo chown -R user:user /app

# Set the default command to python3
# ENTRYPOINT [""]
# CMD ["python3"]