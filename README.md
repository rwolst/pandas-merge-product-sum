# Merge-Product-Sum for Pandas

Attempts to solve the issue of merge-product-sum in Pandas. This is essentially
a sparse matrix multiplication but calling each Pandas function individually is
far less efficient. Instead we convert the Pandas dataframes into sparse Scipy
matrices, perform the matrix multiplication then transform back to dataframes.
See [this SO question][SO question] for a more detailed description.

[SO question]: https://stackoverflow.com/questions/49769774/efficient-merge-product-sum-with-pandas/49770629#49770629.
