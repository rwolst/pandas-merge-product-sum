import pytest
import pandas as pd
import numpy as np
import scipy as sp
import scipy.sparse

from mps import (merge_product_sum, to_sparse, multiply_sparse,
                 reverse_index_map)

@pytest.fixture()
def df1():
    df = pd.DataFrame([
                       [20160404, 'John', 'Z_0', 0.3],
                       [20160404, 'John', 'Z_1', 0.2],
                       [20160404, 'John', 'Z_4', 0.5],
                       [20160404, 'Toby', 'Z_0', 0.4],
                       [20160404, 'Toby', 'Z_1', 0.6],
                       [20160404, 'Toby', 'Z_4', 0.0],
                       [20160407, 'John', 'Z_0', 0.1],
                       [20160407, 'John', 'Z_1', 0.5],
                       [20160407, 'John', 'Z_4', 0.4]
                      ], columns = ['DATE', 'NAME', 'Z', 'PROB'])

    # Assert group by each (DATE, NAME) and summing is 1.
    return df

@pytest.fixture()
def df_large():
    """Create a large random dataframe for testing."""
    # Sample the discrete sets of values (just so they are not np.arange).
    N = 1000
    n_date = 100
    n_name = 100
    n_country = 100
    n_team = 100
    n_z = 4
    DATE_set = np.random.choice(N, n_date, replace=False)
    NAME_set = np.random.choice(N, n_name, replace=False)
    COUNTRY_set = np.random.choice(N, n_country, replace=False)
    TEAM_set = np.random.choice(N, n_team, replace=False)
    Z_set = np.random.choice(N, n_z, replace=False)

    # Sample the actual values and create the dataframe.
    N_1 = 3000
    DATES = np.random.choice(DATE_set, N_1, replace=True)[:,None]
    NAMES = np.random.choice(NAME_set, N_1, replace=True)[:,None]
    Z_1 = np.random.choice(Z_set, N_1, replace=True)[:,None]

    df1 = pd.DataFrame(np.concatenate((DATES, NAMES, Z_1), 1),
                       columns=['DATE', 'NAME', 'Z'])

    ## Create random probs (don't worry about summing to certain values).
    df1['PROB'] = np.random.rand(len(df1))

    N_2 = 10
    COUNTRIES = np.random.choice(COUNTRY_set, N_2, replace=True)[:,None]
    TEAMS = np.random.choice(TEAM_set, N_2, replace=True)[:,None]
    Z_2 = np.random.choice(Z_set, N_2, replace=True)[:,None]

    df2 = pd.DataFrame(np.concatenate((COUNTRIES, TEAMS, Z_2), 1),
                       columns=['COUNTRY', 'TEAM', 'Z'])
    df2['PROB'] = np.random.rand(len(df2))

    return df1, df2

@pytest.fixture()
def df1_sparse():
    """Sparse representation of df1."""
    # Alphabetically:
    #     [20160404, 20160407]
    #     ['John', 'Toby']
    rows = [0, 0, 0, 2, 2, 2, 1, 1, 1]
    cols = [0, 1, 2, 0, 1, 2, 0, 1, 2]
    values = [0.3, 0.2, 0.5, 0.4, 0.6, 0.0, 0.1, 0.5, 0.4]

    m = sp.sparse.coo_matrix((values, (rows, cols)), shape=[4,3])

    return m

@pytest.fixture()
def df2():
    df = pd.DataFrame([
                       ['England', 'Warriors', 'Z_0', 0.4],
                       ['England', 'Warriors', 'Z_1', 0.1],
                       ['England', 'Warriors', 'Z_4', 0.4],
                       ['Scotland', 'Spartans', 'Z_0', 0.6],
                       ['Scotland', 'Spartans', 'Z_1', 0.9],
                       ['Scotland', 'Spartans', 'Z_4', 0.6],
                      ], columns = ['COUNTRY', 'TEAM', 'Z', 'PROB'])

    # Assert group by each Z and summing is 1.
    return df

@pytest.fixture()
def df2_sparse():
    """Sparse representation of df2."""
    # Alphabetically:
    #     ['England', 'Scotland']
    #     ['Spartans', 'Warriors']
    rows = [2, 2, 2, 1, 1, 1]
    cols = [0, 1, 2, 0, 1, 2]
    values = [0.4, 0.1, 0.4, 0.6, 0.9, 0.6]

    m = sp.sparse.coo_matrix((values, (rows, cols)), shape=[4,3])

    return m

@pytest.fixture()
def df():
    """The true merge product sum value."""
    df = pd.DataFrame([
                       [20160404, 'John', 'England', 'Warriors', 0.34],
                       [20160404, 'Toby', 'England', 'Warriors', 0.22],
                       [20160407, 'John', 'England', 'Warriors', 0.25],
                       [20160404, 'John', 'Scotland', 'Spartans', 0.66],
                       [20160404, 'Toby', 'Scotland', 'Spartans', 0.78],
                       [20160407, 'John', 'Scotland', 'Spartans', 0.75],
                      ], columns = ['DATE', 'NAME', 'COUNTRY', 'TEAM',
                                    'PROB'])

    return df

@pytest.fixture()
def df_sparse():
    """Sparse representation of df where rows are (DATE, NAME) and columns
    are (COUNTRY, TEAM)."""
    rows = [0, 2, 1, 0, 2, 1]
    cols = [2, 2, 2, 1, 1, 1]
    values = [0.34, 0.22, 0.25, 0.66, 0.78, 0.75]
    n_rows = 4
    n_cols = 4

    m = sp.sparse.coo_matrix((values, (rows, cols)), shape=[n_rows, n_cols])

    return m

def merge_product_sum_pandas(df1, df2):
    df = pd.merge(df1, df2, on=['Z'])
    df['PROB'] = df['PROB_x']*df['PROB_y']
    df.drop(['PROB_x', 'PROB_y'], axis=1, inplace=True)
    df = df.groupby(['DATE', 'NAME', 'COUNTRY', 'TEAM']).sum()
    df = df.reset_index()

    return df

def test_reverse_index_map(df1, df2, df, df_sparse):
    """Test reverse index on all our sparse matrices."""
    df1_reversed = reverse_index_map(df_sparse.row, 
                                     df1.set_index(['DATE', 'NAME', 'Z']).index)
    df2_reversed = reverse_index_map(df_sparse.col, 
                                     df2.set_index(['COUNTRY', 'TEAM', 'Z']).index)

    assert (df[['DATE', 'NAME']] == df1_reversed).all().all()
    assert (df[['COUNTRY', 'TEAM']] == df2_reversed).all().all()

def test_to_sparse(df1, df1_sparse, df2, df2_sparse):
    """Test the to_sparse function."""
    idx1 = ['DATE', 'NAME']
    idx2 = ['COUNTRY', 'TEAM'] 
    m1 = to_sparse(df1.set_index(idx1 + ['Z']), idx1, ['PROB'])
    m2 = to_sparse(df2.set_index(idx2 + ['Z']), idx2, ['PROB'])

    assert (m1 != df1_sparse).size == 0
    assert (m2 != df2_sparse).size == 0

def test_multiply_sparse(df1_sparse, df2_sparse, df_sparse):
    m = multiply_sparse(df1_sparse, df2_sparse)

    # Numerical error make it hard to use != and ==
    assert ((m.todense() - df_sparse.todense())**2 < 1e-7).all()

def extract_prob(df):
    """Orders and extracts probability from dataframe for testing."""
    col_idx = ['DATE', 'NAME', 'COUNTRY', 'TEAM']
    out = df.sort_values(col_idx).reset_index(drop=True)['PROB']

    return out

def test_merge_product_sum(df1, df2, df):
    mps = merge_product_sum(df1, df2, on=['Z'], lindex=['DATE', 'NAME'],
                            rindex=['COUNTRY', 'TEAM'], lval=['PROB'],
                            rval=['PROB'])

    mps_pandas = merge_product_sum_pandas(df1, df2)

    assert ((extract_prob(mps) - extract_prob(df))**2 < 1e-7).all().all()
    assert ((extract_prob(mps) - extract_prob(mps_pandas))**2 < 1e-7).all().all()

def test_merge_product_sum_large(df_large):
    df1, df2 = df_large
    mps = merge_product_sum(df1, df2, on=['Z'], lindex=['DATE', 'NAME'],
                            rindex=['COUNTRY', 'TEAM'], lval=['PROB'],
                            rval=['PROB'])

    mps_pandas = merge_product_sum_pandas(df1, df2)

    assert ((extract_prob(mps) - extract_prob(mps_pandas))**2 < 1e-7).all().all()
