# swi-ml
A machine learning library written from scratch - with runtime switchable backend!
<p align="center">
  <a href="https://github.com/aitikgupta/swi-ml">
    <img src="logo/swi-ml.png" alt="Logo" width="200" height="100">
  </a>
  <p align="center">
    Provides a single interface to interact with single-core CPU operations (with <a href="https://numpy.org/">NumPy</a> backend), as well as thousands of cores on a GPU (with <a href="https://cupy.dev/">CuPy</a> backend), in runtime!
  </p>
</p>

[![Python][python-shield]][python-url]
[![Code style][black-shield]][black-url]
[![Stargazers][stars-shield]][stars-url]
[![Codecov][codecov-shield]][codecov-url]
[![Personal Website][website-shield]][website-url]
[![MIT License][license-shield]][license-url]

NOTE: This is NOT an alternative to libraries like [scikit-learn](https://scikit-learn.org/) and [cuML](https://docs.rapids.ai/api/cuml/stable/). Their interfaces are complete on their own!


### Prerequsites

[swi-ml](https://github.com/aitikgupta/swi-ml) is built on bare Python and NumPy backbones, all other dependencies are optional!
* [NumPy](https://numpy.org/)
* [CuPy](https://cupy.dev/) _(Optional)_
* [Matplotlib](https://matplotlib.org) _(Optional)_

### Installation

1. _(Optional)_ Setup a virtual environment using `virtualenv` or `anaconda`.
2. Install [NumPy](https://numpy.org/) by following their [insallation guide](https://numpy.org/install/) or simply via `pip`:
    ```console
    pip install numpy
    ```
3. _(Optional)_ For GPU-supported backend, setup a working installation of [CuPy](https://cupy.dev/) by following their [installation guide](https://docs.cupy.dev/en/stable/install.html#install-cupy).
    ```console
    python -c 'import cupy; cupy.show_config()'
    ```
4. _(Optional)_ Install [Matplotlib](https://matplotlib.org) to plot specific curves. (via their [installation guide](https://matplotlib.org/users/installing.html))
5. Install `swi-ml`:
   ```console
   pip install swi-ml  # from PyPI
   pip install git+https://github.com/aitikgupta/swi-ml  # from GitHub
   ```
6. _(Optional)_ To run the pre-defined tests, install [pytest](https://docs.pytest.org/en/stable/) by following their [installation guide](https://docs.pytest.org/en/stable/getting-started.html) or simply via `pip`:
   ```console
   pip install pytest
   ```

## Usage

### Switching backend
```python
from swi_ml import set_backend

# numpy backend (CPU)
set_backend("numpy")

# cupy backend (GPU)
set_backend("cupy")
```

### Automatic fallback
Don't have a physical GPU, or don't know if you have a proper setup for a [GPU-enabled backend](https://github.com/aitikgupta/swi-ml#installation)?

Set automatic fallback (to [NumPy](https://github.com/aitikgupta/swi-ml#installation) - the only hard dependency):
```python
from swi_ml import set_automatic_fallback

# this has been enabled by default for tests
# see https://github.com/aitikgupta/swi-ml/blob/master/tests/__init__.py
set_automatic_fallback(True)
```

### A simple Linear Regression with Gradient Descent
```python
from swi_ml.regression import LinearRegressionGD

data = [[1], [2], [3]]
labels = [2, 4, 6]

model = LinearRegressionGD(
    num_iterations=3,
    learning_rate=0.1,
    normalize=False,
    initialiser="uniform",
    verbose="DEBUG",
)

model.fit(data, labels)

print("Current MSE:", model.curr_loss)
```

#### Output:
```console
INFO: Backend is not set, using default `numpy`
INFO: Setting backend: numpy
INFO: MSE (1/3): 13.93602
INFO: MSE (2/3): 0.22120
INFO: MSE (3/3): 0.05478
INFO: Training time: 0.00035 seconds
Current MSE: 0.054780625247184585
```

_For more concrete examples, please refer to [examples directory](https://github.com/aitikgupta/swi-ml/tree/master/examples)._

### Running the tests

To run the testing suite, execute the following command in the root directory:
```console
python -mpytest  # run the whole suite
python -mpytest tests/test_module.py  # run the specific test module
```

## Contributing

Contributions are what makes the open source community such an amazing place to learn, inspire, and create. Any contributions are _much appreciated!_

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

Distributed under the MIT License. See `LICENSE` for more information.


## Acknowledgements

* Logo created at [LogoMakr.com](https://logomakr.com/9smwTn)
* [Img Shields](https://shields.io)
* [Choose an Open Source License](https://choosealicense.com)

## About

Aitik Gupta - [Personal Website](aitikgupta.github.io)

[black-shield]: https://img.shields.io/badge/code%20style-black-000000.svg?style=flat-square
[black-url]: https://github.com/psf/black
[contributors-shield]: https://img.shields.io/github/contributors/aitikgupta/swi-ml.svg?style=flat-square
[contributors-url]: https://github.com/aitikgupta/swi-ml/graphs/contributors
[codecov-shield]: https://img.shields.io/codecov/c/gh/aitikgupta/swi-ml?style=flat-square
[codecov-url]: https://codecov.io/gh/aitikgupta/swi-ml
[forks-shield]: https://img.shields.io/github/forks/aitikgupta/swi-ml.svg?style=flat-square
[forks-url]: https://github.com/aitikgupta/swi-ml/network/members
[stars-shield]: https://img.shields.io/github/stars/aitikgupta/swi-ml.svg?style=flat-square
[stars-url]: https://github.com/aitikgupta/swi-ml/stargazers
[issues-shield]: https://img.shields.io/github/issues/aitikgupta/swi-ml.svg?style=flat-square
[issues-url]: https://github.com/aitikgupta/swi-ml/issues
[license-shield]: https://img.shields.io/github/license/aitikgupta/swi-ml.svg?style=flat-square
[license-url]: https://github.com/aitikgupta/swi-ml/blob/master/LICENSE
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=flat-square&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/aitik-gupta
[product-screenshot]: images/screenshot.png
[python-shield]: https://img.shields.io/badge/python-3.7+-blue.svg
[python-url]: https://www.python.org/
[website-shield]: https://img.shields.io/badge/website-aitikgupta.ml-blue?style=flat-square
[website-url]: https://aitikgupta.github.io/