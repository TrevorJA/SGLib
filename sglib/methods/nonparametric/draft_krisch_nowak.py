import numpy as np
import pandas as pd
import warnings
from sglib.core.base import Generator


class KirschNowakGenerator(Generator):

    def __init__(self, Q: pd.DataFrame, **kwargs):
        if not isinstance(Q, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame.")
        if not isinstance(Q.index, pd.DatetimeIndex):
            raise TypeError("Input index must be a pd.DatetimeIndex.")

        self.Q = Q.copy()
        self.params = {
            'generate_using_log_flow': True,
            'matrix_repair_method': 'spectral',
            'debug': False,
        }
        self.params.update(kwargs)
        self.fitted = False
        
        self.n_months = 12  # Number of months in a year

    def preprocessing(self, timestep='monthly'):
        # Group the daily data into monthly totals
        if timestep == 'monthly':
            monthly = self.Q.groupby([self.Q.index.year, self.Q.index.month]).sum()
            monthly.index = pd.MultiIndex.from_tuples(monthly.index, names=['year', 'month'])
            self.Qm = monthly.unstack(level=-1)
        else:
            raise NotImplementedError("Currently only monthly timestep is supported.")

        # Optionally apply log transformation to the flows
        if self.params['generate_using_log_flow']:
            self.Qm = np.log(self.Qm.clip(lower=1e-6))

    def fit(self):
        # Calculate the mean and standard deviation for each month
        self.mean_month = self.Qm.mean(axis=0)
        self.std_month = self.Qm.std(axis=0)

        # Standardize the historical flows
        Z_h = (self.Qm - self.mean_month) / self.std_month
        self.Z_h = Z_h.dropna()

        # Compute intraannual and interannual correlation matrices
        self.corr_intra = self.Z_h.corr()
        self.corr_inter = self._build_interyear_corr(self.Z_h)

        # Perform Cholesky decompositions on correlation matrices
        self.U = self._repair_and_cholesky(self.corr_intra)
        self.U_prime = self._repair_and_cholesky(self.corr_inter)

        self.historic_years = self.Z_h.index.get_level_values(0).unique()
        self.n_years = len(self.historic_years)
        self.n_months = self.Z_h.shape[1]

        self.fitted = True

    def generate_single_series(self, n_years):
        # Bootstrap resampling of historic years
        M = np.random.choice(self.historic_years, size=(n_years + 1, self.n_months), replace=True)
        C = np.vstack([self.Z_h.loc[year].values for year in M[:, :]])

        # Apply Cholesky transformations
        Z_c, Z_c_prime = self._transform_C(C)

        # Merge intra- and interannual correlations
        Z_final = self._combine_interannual(Z_c, Z_c_prime)

        # De-standardize the synthetic flows
        Q_syn = (Z_final * self.std_month.values) + self.mean_month.values

        if self.params['generate_using_log_flow']:
            Q_syn = np.exp(Q_syn)

        return Q_syn

    def generate(self, n_realizations=1, n_years=None):
        if not self.fitted:
            raise RuntimeError("Call preprocessing() and fit() before generate().")

        n_years = n_years or self.n_years
        reals = [self.generate_single_series(n_years) for _ in range(n_realizations)]

        return np.stack(reals, axis=0)  # (n_realizations, n_years * 12)

    ####### Helper Methods ########

    def _repair_and_cholesky(self, A, 
                                max_iter = 20, 
                                debugging = True,
                                method = 'spectral'):
        """
        Attempts to perform cholskey decomp, and repairs as needed. 

        Method: https://www.mathworks.com/matlabcentral/answers/6057-repair-non-positive-definite-correlation-matrix

        A : matrix to be decomposed
        """
        machine_eps = np.finfo(float).eps

        pass_test = self._check_positive_definite(A)
        if pass_test:
            return np.linalg.cholesky(A)
        
        elif not pass_test:
            if debugging:
                print('Matrix is not posdef. Attempting repair.')
            i = 0
        
            while not pass_test and (i < max_iter):
                # Get eigenvalues
                L, V = np.linalg.eig(A)
                real_lambdas = np.real(L)

                # Find the minimum eigenvalue
                neg_lambda = min(real_lambdas)
                neg_lambda_index = np.argmin(real_lambdas)
                neg_V = V[:, neg_lambda_index]
                if debugging:
                    print(f'Negative lambda ({neg_lambda}) in column {neg_lambda_index}.')

                if method == "spectral":
                    # Remove negative eigenvalues
                    pos_L = np.where(L > 0, L, 0)
                    
                    # Reconstruct matrix
                    A = V @ np.diag(pos_L) @ V.T
                
                elif method == 'rank':
                    # Add machine epsilon
                    shift = np.spacing(neg_lambda) - neg_lambda
                    A = A + neg_V* neg_V.T * shift
                
                elif method == 'custom_diag':
                    diff = min([neg_lambda, -np.finfo(float).eps])
                    shift = diff * np.eye(A.shape[0])
                    A = A - shift
            
                elif method == 'simple_diag':    
                    A = A + machine_eps * np.eye(len(A))

                else:
                    print('Invalid repair method specified.')
                    return

                pass_test = self._check_positive_definite(A)
                
                i += 1

            if pass_test:
                if debugging:
                    print(f'Matrix repaired after {i} updates.')
                return np.linalg.cholesky(A)
            elif not pass_test:
                print(f'Matrix could not repaired after {i} updates!')
                return 

    def _check_positive_definite(self, A):
        try:
            np.linalg.cholesky(A)
            return True
        except np.linalg.LinAlgError:
            return False

    def _build_interyear_corr(self, Z_h):
        # Build a matrix capturing interyear correlations (half-year offset)
        half_shift = self.n_months // 2
        Z_flat = Z_h.values.flatten()
        Z_shifted = Z_flat[half_shift:]
        Z_main = Z_flat[:-half_shift]
        corr = np.corrcoef(np.vstack([Z_main, Z_shifted]))
        return corr

    def _transform_C(self, C):
        # Apply intraannual and interannual transformations
        C_full = C[:-1, :]
        C_prime = C[1:, :]

        Z = C_full @ self.U
        Z_prime = C_prime @ self.U_prime

        return Z, Z_prime

    def _combine_interannual(self, Z, Z_prime):
        # Combine Z and Z' to build final series preserving interannual correlation
        n, t = Z.shape
        Z_combined = np.zeros((n, t))

        half = t // 2
        Z_combined[:, :half] = Z_prime[:, half:]
        Z_combined[:, half:] = Z[1:, :half]

        return Z_combined
