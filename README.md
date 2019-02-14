# Kernel ISOMAP

A RBF-Kernelized version of ISOMAP (Isometric Mapping). A description of the algorithm can be found [here](https://scikit-learn.org/stable/modules/manifold.html#isomap).

## Prerequisites

* [NumPy](http://www.numpy.org/) - NumPy is the fundamental package for scientific computing with Python.
* [scikit-learn](https://scikit-learn.org/stable/) - Machine Learning in Python

### Python
The code is compatible with python 3.x .

## Example

```
from sklearn.datasets import load_digits
from kernel_isomap import *

if __name__ == '__main__':
    digits = load_digits(n_class=6)
    X = digits.data
    X_embedded = KernelIsomap(n_jobs=-1).fit_transform(X)
    print(X_embedded.shape)
```

## Author

**Antonio Roberto** - [Linkedin](https://www.linkedin.com/in/antonio-roberto-1b288b120/)

## License

This project is licensed under the GNU GENERAL PUBLIC LICENSE - see the [LICENSE.md](LICENSE.md) file for details