# NuNNERY - Numeric Neural-Network Extensible Research Youngster 
Every projects needs a punny name - coming up with one *has* to be at least 30% of the development budget. In lieu of an actual budget, I spent about 5 minutes on this name. Yes, it shows... I know. ;)

## Purpose - Scope - Progress
NuNNERY is currently a script which implements a fully-connected neural network with mini-batch backpropagation-learning for MNIST-classification from scratch. It is intended for purposes of demonstration, learning, and experimentation. Production-grade performance is not within scope.

In the next iteration, NuNNERY will become a fully typed, extensible and testable solution in mixed-paradigm programming (object-oriented with functional and imperative programming used within methods). It will demonstrate how to apply principles of Domain-Driven-Design and *SOLID* programming to python 3.  Additionally, it will feature live-presentation of the learning progress using matplotlib's pyplot. TKinter will be used for interactive drawing of the network's input after training and testing.

Progress towards this next iteration can be viewed in the `mixed-paradigm` branch.

## Architectural Principles
Mixed-paradigm code is used throughout the project. The general structure is object-oriented according to principles of Domain-Driven-Design and SOLID programming with fully typed signatures using Generics where helpful. Inside classes and methods, both procedural and (limited) functional programming is used.

The core domain is represented in the `src/core` subdirectory. Code for learning is in `src/learning`, for visualization in `src/visualization` and for interactivity in `src/interactivity`. The `src` directory also contains the main entry point for the application.

## Used libraries
* `typing` for type annotations
* `numpy` for matrix operations
* `opencv-python` for displaying and manipulating images
* `scipy.ndimage` for manipulating images as ndarrays
* `keras.datasets` for loading MNIST data (Requires large install tensorflow - you can avoid this by providing your own data)
* `pickle` for saving and loading trained networks
* `math` for math functions
* `lzma` for compressing transformed data to store as file for faster startup in future executions

## Installation

### Using Conda
1. Install [Anaconda](https://www.anaconda.com/distribution/) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html)
2. Create a new environment with Python 3.11: `conda create -n nunnery python=3.11`
3. Activate the environment: `conda activate nunnery`
4. Install the dependencies: `conda install numpy opencv-python scipy keras`
5. Clone the repository: `git clone`
6. Run the code (this assumes that `python3` is used): `python3 nunnery.py`
