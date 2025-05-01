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

    def preprocessing(self, timestep='monthly'):
        # Group the daily data into monthly totals
        if timestep == 'monthly':
            monthly = self.Q.groupby([self.Q.index.year, self.Q.index.month]).sum()
            monthly.index = pd.MultiIndex.from_tuples(monthly.index, names=['year', 'month'])
            self.Qm = monthly.unstack(level=-1)
        else:
            raise NotImplementedError("Currently only monthly timestep is supported.")

        if self.params['generate_using_log_flow']:
            self.Qm = np.log(self.Qm.clip(lower=1e-6))

    def fit(self):
        # Compute monthly means and stds for standardization
        self.mean_month = self.Qm.mean(axis=0)
        self.std_month = self.Qm.std(axis=0)

        # Standardize historic flows
        Z_h = (self.Qm - self.mean_month) / self.std_month
        self.Z_h = Z_h.dropna()

        # Build intraannual correlation matrix
        self.corr_intra = self.Z_h.corr()

        # Build interannual correlation matrix using correct year shifting
        self.corr_inter = self._build_interyear_corr(self.Z_h)

        # Decompose correlation matrices
        self.U = self._repair_and_cholesky(self.corr_intra)
        self.U_prime = self._repair_and_cholesky(self.corr_inter)

        self.historic_years = self.Z_h.index.get_level_values(0).unique()
        self.n_years = len(self.historic_years)
        self.n_sites = self.Q.shape[1]
        self.n_months = 12

        self.fitted = True

    def _get_random_bootstrap(self, n_years):
        M = np.random.choice(self.historic_years, size=(n_years+2,), replace=True)
        return M

    def generate_single_series(self, n_years, M=None):
        
        if M is None:
            # Bootstrap years
            M = self._get_random_bootstrap(n_years)
        else:
            # Check if M is a valid bootstrap year array
            if not isinstance(M, (list, np.ndarray)):
                raise TypeError("M must be a list or numpy array.")
            if len(M) != n_years + 2:
                raise ValueError("Length of M must be n_years + 2.")

        
        # Create bootstrap matrix C
        C = np.vstack([self.Z_h.loc[year].values for year in M])

        # Apply intraannual correlation
        Z = (C[:-1, :] @ self.U)
        Z_prime = (C[1:, :] @ self.U_prime)

        # Combine Z and Z_prime properly across mid-year split
        Z_combined = self._combine_interannual(Z, Z_prime)

        # De-standardize
        Q_syn = (Z_combined * self.std_month.values) + self.mean_month.values

        if self.params['generate_using_log_flow']:
            Q_syn = np.exp(Q_syn)

        # Reshape to (sites, full timeseries length)
        Q_syn = Q_syn.reshape((n_years, self.n_sites, self.n_months))
        Q_syn = Q_syn.transpose(1, 0, 2).reshape(self.n_sites, n_years * self.n_months)

        return Q_syn

    def generate(self, n_realizations=1, n_years=None):
        if not self.fitted:
            raise RuntimeError("Call preprocessing() and fit() before generate().")

        n_years = n_years or self.n_years
        reals = [self.generate_single_series(n_years) for _ in range(n_realizations)]

        return np.stack(reals, axis=0)

    ####### Helper Methods ########

    def _check_positive_definite(self, matrix):
        """Check if a matrix is positive definite."""
        try:
            np.linalg.cholesky(matrix)
            return True
        except np.linalg.LinAlgError:
            return False

    def _repair_and_cholesky(self, corr):
        try:
            return np.linalg.cholesky(corr).T
        except np.linalg.LinAlgError:
            warnings.warn("Matrix not positive definite, repairing...")
            non_psd = True
            iter = 0
            while non_psd and iter < 50:
                evals, evecs = np.linalg.eigh(corr)
                evals[evals < 0] = 1e-8
                corr = evecs @ np.diag(evals) @ evecs.T
                iter += 1
                try:
                    np.linalg.cholesky(corr)
                    non_psd = False
                except np.linalg.LinAlgError:
                    non_psd = True
                    
            if non_psd:
                warnings.warn("Matrix could not be repaired to be positive definite.")
                raise ValueError("Matrix is not positive definite after 50 iterations.")
            else:
                print(f"Matrix repaired successfully after {iter} iterations.")
                return np.linalg.cholesky(corr).T


    def _build_interyear_corr(self, Z_h):
        n_years = Z_h.index.get_level_values(0).nunique()
        n_vars = Z_h.shape[1]

        Z_mat = Z_h.values.reshape((n_years, n_vars))
        Z_main = Z_mat[:-1, :]
        Z_shifted = Z_mat[1:, :]

        # Direct correlation between year t and year t+1
        corr = np.corrcoef(Z_main.T, Z_shifted.T)[:n_vars, n_vars:]

        return corr

    def _combine_interannual(self, Z, Z_prime):
        n, t = Z.shape
        half = t // 2

        Z_combined = np.zeros((n-1, t))

        # First half of each synthetic year from second half of Z'
        Z_combined[:, :half] = Z_prime[:-1, half:]

        # Second half from first half of Z
        Z_combined[:, half:] = Z[1:, :half]

        return Z_combined
