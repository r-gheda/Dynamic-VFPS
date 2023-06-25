# Dynamic-VFPS
### Distributed ML Seminar Project 2022/23

## How to run
To run:

1. Build the image with `docker build -t pyvertical:latest .`
1. Launch a container with `docker run -it -p 8888:8888 pyvertical:latest`
    - Defaults to launching jupyter lab

## Code design

This framework design took inspiration from the [PySyft](https://blog.openmined.org/tag/pysyft/)-based [PyVertical](https://github.com/OpenMined/PyVertical) framework by OpenMined. The original framework supported SplitNN with one client owning all the data and a server owning all the labels. PyVertical have been extended with new functionalities, including:
- Multi-client Vertical Federated Learning support (Server owns labels, clients own data)
- Dynamic VFL system support (More details below)
- Some minor dependencies fixes for Python packages' requirements

## Contributions

This work is an improvement on paper: "[VF-PS: How to Select Important Participants in Vertical Federated Learning, Efficiently and Securely?](https://openreview.net/forum?id=vNrSXIFJ9wz)".
Further details are provided in the [report](Report.pdf).
