import scipy as sp
import scipy.sparse
import numpy as np
import pandas as pd


def merge_product_sum(df1, df2, on, lindex, rindex, lval, rval):
    # Set index on dataframes
    df1 = df1.set_index(lindex + on)
    df2 = df2.set_index(rindex + on)

    ## Must have the Z labels the same
    Z_levels_1 = set(df1.index.levels[-1])
    Z_levels_2 = set(df2.index.levels[-1])
    Z_levels = list(Z_levels_1.union(Z_levels_2))

    ## Match the levels.
    perm = pd.core.algorithms.match(df1.index.levels[-1], Z_levels)
    df1.index.set_levels(Z_levels, 'Z', inplace=True)
    df1.index.set_labels(perm[df1.index.labels[-1]], 'Z', inplace=True)

    perm = pd.core.algorithms.match(df2.index.levels[-1], Z_levels)
    df2.index.set_levels(Z_levels, 'Z', inplace=True)
    df2.index.set_labels(perm[df2.index.labels[-1]], 'Z', inplace=True)

    ## Add on to the end of the Z levels any unseen Z values in both
    ## dataframes.
    assert (df1.index.levels[-1] == df2.index.levels[-1]).all()

    # Create sparse matrix from both dataframes
    m1 = to_sparse(df1, lindex, lval)
    m2 = to_sparse(df2, rindex, rval)

    # Convert to csr and multiply the matrices
    m = multiply_sparse(m1, m2)

    # Convert back to dataframe
    df = from_sparse(m, df1, df2)

    return df


def from_sparse(m, df1, df2):
    """Convert sparse matrix back to a Pandas dataframe."""
    # First split the rows back into labels on df1
    names1 = reverse_index_map(m.row, df1.index)
    names2 = reverse_index_map(m.col, df2.index)
    values = pd.DataFrame(m.data, columns=['PROB'])

    df = pd.concat((names1, names2, values), axis=1)

    return df


def reverse_index_map(x, idx):
    """Reverses the mapping of the df multiindex onto integers.
       Input:
           d: A (len(df), ) vector of mapped integers.
           idx: Dataframe index.
       Output:
           df: Dataframe with the multiindex rest as columns.
    """
    # Get multipliers vector required for reversing the mapping.
    n_idx = len(idx.levels) - 1
    multipliers = np.cumprod([1] + [idx.levels[i].size
                                    for i in range(n_idx)])

    # Store the values intermediately in a dictionary.
    values = {}
    for i in range(n_idx-1,-1,-1):
        mult = multipliers[i]
        name = idx.names[i]

        d, r = np.divmod(x, mult)

        values[name] = idx.levels[i][d]
        x = r

    out = pd.DataFrame(values, columns=idx.name)

    return out


def to_sparse(df, index, val):
    mult = 1
    rows = np.zeros(len(df))
    for i in range(len(index)):
        rows = rows + mult*df.index.labels[i].astype(np.int64)
        mult = mult*df.index.levels[i].size
    n_rows = mult

    cols = df.index.labels[-1]
    n_cols = df.index.levels[-1].size
    # Currently only do it with 1 value
    assert len(val) == 1
    values = df[val[0]]

    return sp.sparse.coo_matrix((values, (rows, cols)), shape=[n_rows, n_cols])


def multiply_sparse(m1, m2):
    """Take sparse matrices m1 and m2 in coo format, calculate
        m1 @ m2.T
    using csr format and then convert back to coo format.
    """
    return (m1.tocsr() @ m2.tocsr().T).tocoo()
