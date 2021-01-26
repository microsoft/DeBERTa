# This is the docker file built for GLUE task
FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu18.04
RUN apt-get update && apt-get install -y vim gawk wget software-properties-common vim-addon-manager locales gdb 
RUN apt-get update && apt-get install -y python3.6 python3.6-dev python3.6-doc python3.6-dbg 
#python-distutils
ENV  LANG="en_US.UTF-8" LC_ALL="en_US.UTF-8" LANGUAGE="en_US.UTF-8" LC_TYPE="en_US.UTF-8" TERM=xterm-256color
RUN locale-gen en_US en_US.UTF-8

RUN apt-get update
RUN apt-get install -y git build-essential
RUN apt-get install -y rsync jq sudo net-tools apt-transport-https curl ca-certificates clang libglib2.0-dev
RUN apt-get install -y htop bash-completion libibverbs-dev graphviz libunwind-dev

RUN apt-get install -y openjdk-8-jdk && \
    echo "deb [arch=amd64] http://storage.googleapis.com/bazel-apt stable jdk1.8" | sudo tee /etc/apt/sources.list.d/bazel.list && \
    curl https://bazel.build/bazel-release.pub.gpg | sudo apt-key add - && \
	  apt-get update 

RUN 	curl https://bootstrap.pypa.io/get-pip.py | python3.6 && \
 	update-alternatives --install /usr/bin/pdb3 pdb3 /usr/bin/pdb3.6 2 && \
 	update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.6 2 && \
	update-alternatives --install /usr/bin/python python /usr/bin/python2.7 1 && \
	update-alternatives --install /usr/bin/python python /usr/bin/python3.6 2 && \
	pip install ipython==7.5.0 pycparser

RUN pip3 install --upgrade setuptools

WORKDIR /work
COPY requirements.txt /work/requirements.txt
RUN pip install -r /work/requirements.txt

#RUN pip3 install -q beautifulsoup4 progressbar2 nltk==3.4.1 html2text

RUN python -c "import nltk;nltk.download('punkt')"

#RUN pip3 install -q joblib tqdm ftfy spacy==2.1.4
RUN python -m spacy download en

#RUN pip3 install numpy==1.16.3 scipy==1.2.1 matplotlib jupyter pandas==0.24.2 sympy==1.4 nose==1.3.7 sklearn 
#RUN pip3 install torch===1.3.1 torchvision===0.4.0 -f https://download.pytorch.org/whl/torch_stable.html

#RUN pip3 install ujson cython xmltodict pytest


#RUN pip3 install regex psutil boto3 sentencepiece newspaper3k seqeval

