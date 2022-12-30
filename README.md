# DroNET - Autonomous Networking 2022-2023

## What is DroNET?
DroNET is a Python based simulator for experimenting routing algorithms and mobility models on unmanned aerial vehicle 
networks. 

# Setup
The project is developed using Python 3.10. To install the required packages, you can use the provided conda environment:

```bash
conda env create -f environment.yml
```


## Execution

In order to start the simulator you can run the following command:

```bash
conda activate droNET
python -m src.main
```

## Simulator structure 
The project has the following structure:

The entry point of the project is the ``src.main`` file, from there you can run simulations and extensive
 experimental campaigns, by setting up an appropriate ``src.simulator.Simulator`` object. 
 
The two main directories are ``data`` and ``src``. The directory ``data``  contains all the 
data of the project, like drones tours, and other input and output of the project. The directory ``src`` 
contains the source code, organized in several packages. 

* ``src.drawing`` it contains all the classes needed for drawing the simulation on screen. Typically, you may 
want to get your hands in this directory if you want to change the aspect of the simulation, display a new 
object, or label on the area.

* ``src.entites`` it contains all the classes that define the behaviour and the structure of the main
 entities of the project like: Drone, Depot, Environment, Packet, Event classes.

* ``src.experiments`` it contains classes that handle experimental campaigns.

* ``src.plots`` it contains classes to perform plotting operations 

* ``src.routing_algorithms`` it contains all the classes modelling the several routing algorithms, 
**every** routing algorithm should have its own class.

* ``src.simulation`` it contains all the classes to handle a simulation and its metrics. 

* ``src.utilities`` it contains all the utilities and the configuration parameters. In particular use ``src.utilities.config`` file to 
specify all the constants and parameters for a one-shot simulation, ideal when one wants to evaluate
the quality of a routing algorithm making frequent executions. Constants and parameters should **always** be added here
and never be hard-coded.

## Thanks and License
The current version of the simulator is free for non-commercial use.

This project is based on the paper [A Q-Learning-Based Topology-Aware Routing Protocol for Flying Ad Hoc Networks](https://ieeexplore.ieee.org/document/9456858)

The simulator was done by Andrea Coletta in collaboration with Matteo Prata, PhD Student at La Sapienza  **coletta[AT]di.uniroma1.it**, **prata[AT]di.uniroma1.it** and [later extended](https://github.com/flaat/DroNETworkSimulator) by Flavio Giorgi  **flavio.giorgi[AT]uniroma1.it** and Giulio Attenni **giulio.attenni[AT]uniroma1.it**.