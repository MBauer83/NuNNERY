# NuNNERY - Numeric Neural-Network Extensible Research Youngster 
Every projects needs a punny name - coming up with that *has* to be at least 30% of the development budget. In lieu of an actual budget, I spent about 5 minutes on this one. Yes, it shows... I know. ;)

## Purpose and Scope
NuNNERY is a simple, easy to use, and extendable library for learning and experimenting with neural networks. It is written in Python 3. It demonstrates the structured implementation of a network-architecture and a backpropagation-learning algorithm as well as live-presentation of the learning progress using matplotlib's pyplot. TKinter is used for interactive drawing of the network's input after training and testing. 

## Architectural Principles
Mixed-paradigm code is used throughout the project. The general structure is object-oriented according to principles of Domain-Driven-Design and SOLID programming with fully typed signatures using Generics where helpful. Inside classes and methods, both procedural and (limited) functional programming is used.

The core domain is represented in the `src/core` subdirectory. Code for learning is in `src/learning`, for visualization in `src/visualization` and for interactivity in `src/interactivity`. The `src` directory also contains the main entry point for the application.

## Used libraries
* `typing` for type annotations
* `numpy` for matrix operations
* `matplotlib` for visualization
* `tkinter` for interactivity
* `keras.datasets` for loading MNIST data (Requires large install tensorflow - you can avoid this by providing your own data)
* `pickle` for saving and loading trained networks
* `math` for math functions
* `sciPy.ndimage` for transforming MNIST learning data to generate variants

## Installation

### Using Conda
1. Install [Anaconda](https://www.anaconda.com/distribution/) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html)
2. Create a new environment with Python 3.11: `conda create -n nunnery python=3.11`
3. Activate the environment: `conda activate nunnery`
4. Install the dependencies: `conda install numpy matplotlib scipy keras`
5. Clone the repository: `git clone`
6. Select a `DataProvider` from `src/examples/data_providers` or create your own according to the interface in `src/learning/DataProvider.py`
7. Optionally select a `NetworkInteractor` (like `src/interactivity/MNISTDigitDrawingNetworkInteractor.py`) to use after training and testing
7. Execute the `main.py`, providing ive a the location of a `DataProvider` in a parameter `-d` and optionally the location of  `NetworkInteractor` in a parameter `-i`, for example: `python main.py -d src/examples/data_providers/MNISTDataProvider.py -i src/interactivity/MNISTDigitDrawingNetworkInteractor.py`