import subprocess
import datetime
import sys

def install(package):
  subprocess.check_call([sys.executable, "-m", "pip", "install", package])

install('setuptools')
install('gitpython')

import setuptools
import git

repo = git.Repo(search_parent_directories=True)
date=datetime.datetime.utcnow()

with open("README.md", "r") as fh:
  long_description = fh.read() + f'\n git version: {repo.head.object.hexsha}' + \
  f'\n date: {date}'

with open('VERSION') as fs:
    version = fs.readline().strip()
#    version += f".dev{date.strftime('%y%m%d%H%M%S')}"

with open('requirements.txt') as fs:
    requirements = [l.strip() for l in fs if not l.strip().startswith('#')]

extras = {}
extras["docs"] = ["recommonmark", "sphinx", "sphinx-markdown-tables", "sphinx-rtd-theme"]

setuptools.setup(
    name="DeBERTa",
    version=version,
    author="penhe",
    author_email="penhe@microsoft.com",
    description="Decoding enhanced BERT with Disentangled Attention",
    keywords="NLP deep learning transformer pytorch Attention BERT RoBERTa DeBERTa",
    license="MIT",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/microsoft/DeBERTa",
    packages=setuptools.find_packages(exclude=['__pycache__']),
    package_dir = {'DeBERTa':'DeBERTa'},
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    extras_require=extras,
    install_requires=requirements)
