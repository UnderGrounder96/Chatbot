# Chatbot

This project is a Python Chatbot program.<br />
Using Deep Learning, this bot is able to provide answers to simple questions.

## Getting Started

This Python program was created under Ubuntu 20.04.1 LTS Operative System,<br />
Python 3.8.2, Pip 20.0.2, Numpy 1.19.2, Nltk 3.5.0 and Torch 1.6.0.

## Prerequisites

i) Installing Python v3.8+<br />
It is possible that Python has been already installed, one could check by using the commands:

    $ python3 --version
    [Python 3.8.2]

    If errors occurred or Python has not yet been installed use the following code:

    # Refreshes the repositories
    $ sudo apt update

    # Installs latest available python3
    $ sudo apt install -y python3

ii) Installing Pip v20+<br />
In order to install Pip, one could should use the commands:

    $ python3 -m pip --version
    [pip 20.2.2 from ... (python 3.8)]

    # Installing pip
    $ curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
    $ python3 get-pip.py

    # Its possible to update pip using
    $ python3 -m pip install -U pip

    # Install all requried pip modules
    $ pip install -r requirements.txt

iii) Download nltk 'all'<br />
While in interactive mode or using the command:

    $ python3 -c "import nltk; nltk.download('all')"
    [...]

iv) Manually install Nvidia[CUDA devkit](https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&target_distro=Ubuntu&target_version=2004&target_type=debnetwork).

## Deployment

TBA

## Versioning

Version 0.2 - Current version<br />

## Author

Lucio Afonso

## License

This project is licensed under the GPL License - see the LICENSE.md file for details

## Acknowledgments

Official sites:

https://www.python.org/<br />
https://github.com/pypa/pip<br />
