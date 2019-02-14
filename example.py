# Author: Antonio Roberto -- <https://github.com/antonioroberto1994>
# License: GNU General Public License v3.0

from sklearn.datasets import load_digits
from kernel_isomap import *

if __name__ == '__main__':
    digits = load_digits(n_class=6)
    X = digits.data
    X_embedded = KernelIsomap(n_jobs=-1).fit_transform(X)
    print(X_embedded.shape)
