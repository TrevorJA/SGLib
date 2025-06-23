import numpy as np
import pandas as pd
import warnings

from sglib.core.base import Generator

class KirschGenerator(Generator):
    def __init__(self, Q: pd.DataFrame, generate_using_log_flow=True, matrix_repair_method='spectral', debug=False):
        if not isinstance(Q, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame.")
        if not isinstance(Q.index, pd.DatetimeIndex):
            raise TypeError("Input index must be a pd.DatetimeIndex.")

        self.Q = Q.copy()
        self.params = {
            'generate_using_log_flow': generate_using_log_flow,
            'matrix_repair_method': matrix_repair_method,
            'debug': debug,
        }

        self.fitted = False
        self.n_historic_years = Q.index.year.nunique()
        self.n_sites = Q.shape[1]
        self.site_names = Q.columns.tolist()
        self.n_months = 12

        self.U_site = {}
        self.U_prime_site = {}

    def _get_synthetic_index(self, n_years):
        return pd.date_range(start=f"{self.Q.index.year.max() + 1}-01-01", periods=n_years * self.n_months, freq='MS')

    def preprocessing(self, timestep='monthly'):
        if timestep != 'monthly':
            raise NotImplementedError("Currently only monthly timestep is supported.")

        monthly = self.Q.groupby([self.Q.index.year, self.Q.index.month]).sum()
        monthly.index = pd.MultiIndex.from_tuples(monthly.index, names=['year', 'month'])
        self.Qm = monthly

        if self.params['generate_using_log_flow']:
            self.Qm = np.log(self.Qm.clip(lower=1e-6))



    def fit(self):
        self.mean_month = self.Qm.groupby(level='month').mean()
        self.std_month = self.Qm.groupby(level='month').std()

        years = self.Qm.index.get_level_values('year').unique()
        Z_h = []
        valid_years = []

        for year in years:
            try:
                year_data = []
                for m in range(1, 13):
                    row = ((self.Qm.loc[(year, m)] - self.mean_month.loc[m]) / self.std_month.loc[m]).values
                    year_data.append(row)
                Z_h.append(year_data)
                valid_years.append(year)
            except KeyError:
                continue

        self.Z_h = np.array(Z_h)  # shape: (n_years, 12, n_sites)
        self.historic_years = np.array(valid_years)
        self.n_historic_years = len(valid_years)

        self.Y = self.Z_h.copy()
        self.Y_prime = np.zeros_like(self.Y[:-1])
        self.Y_prime[:, :6, :] = self.Y[:-1, 6:, :]
        self.Y_prime[:, 6:, :] = self.Y[1:, :6, :]

        for s in range(self.n_sites):
            y_s = self.Y[:, :, s]           # shape: (n_years, 12)
            y_prime_s = self.Y_prime[:, :, s]

            corr_s = np.corrcoef(y_s.T)     # shape: (12, 12)
            corr_prime_s = np.corrcoef(y_prime_s.T)

            self.U_site[s] = self._repair_and_cholesky(corr_s)
            self.U_prime_site[s] = self._repair_and_cholesky(corr_prime_s)

        self.fitted = True

    def _repair_and_cholesky(self, corr):
        try:
            return np.linalg.cholesky(corr).T
        except np.linalg.LinAlgError:
            repaired_corr = corr.copy()
            print("WARNING: Matrix not positive definite, repairing... This may cause correlation inflation.")
            for _ in range(50):
                evals, evecs = np.linalg.eigh(repaired_corr)
                evals[evals < 0] = 1e-8
                repaired_corr = evecs @ np.diag(evals) @ evecs.T
                np.fill_diagonal(repaired_corr, 1.0)
                try:
                    return np.linalg.cholesky(repaired_corr).T
                except np.linalg.LinAlgError:
                    continue
            raise ValueError("Matrix is not positive definite after 50 iterations.")

    def _get_bootstrap_indices(self, n_years, max_idx=None):
        max_idx = self.n_historic_years if max_idx is None else max_idx
        return np.random.choice(max_idx, size=(n_years, self.n_months), replace=True)

    def _create_bootstrap_tensor(self, M, use_Y_prime=False):
        source = self.Y_prime if use_Y_prime else self.Y
        n_years, n_months = M.shape
        max_idx = source.shape[0]
        output = np.zeros((n_years, n_months, self.n_sites))
        for i in range(n_years):
            for m in range(n_months):
                h_idx = M[i, m]
                if h_idx >= max_idx:
                    h_idx = max_idx - 1
                output[i, m] = source[h_idx, m]
        return output

    def _combine_Z_and_Z_prime(self, Z, Z_prime):
        n_years = min(Z.shape[0], Z_prime.shape[0]) - 1
        ZC = np.zeros((n_years, self.n_months, self.n_sites))
        ZC[:, :6, :] = Z_prime[:n_years, 6:, :]
        ZC[:, 6:, :] = Z[1:n_years+1, :6, :]
        return ZC

    def _destandardize_flows(self, Z_combined):
        Q_syn = np.zeros_like(Z_combined)
        for m in range(self.n_months):
            Q_syn[:, m, :] = Z_combined[:, m, :] * self.std_month.iloc[m].values + self.mean_month.iloc[m].values
        return Q_syn

    def _reshape_output(self, Q_syn):
        return Q_syn.reshape(-1, self.n_sites)

    def generate_single_series(self, n_years, M=None, as_array=True, synthetic_index=None):
        n_years_buffered = n_years + 1
        if not self.fitted:
            raise RuntimeError("Call preprocessing() and fit() before generate().")

        if M is None:
            M = self._get_bootstrap_indices(n_years_buffered, max_idx=self.Y.shape[0])
        else:
            M = np.asarray(M)
            if M.shape != (n_years_buffered, self.n_months):
                raise ValueError(f"M must have shape ({n_years_buffered}, {self.n_months})")

        M_prime = M[:self.Y_prime.shape[0], :]

        X = self._create_bootstrap_tensor(M, use_Y_prime=False)
        X_prime = self._create_bootstrap_tensor(M_prime, use_Y_prime=True)

        Z = np.zeros_like(X)
        Z_prime = np.zeros_like(X_prime)

        for s in range(self.n_sites):
            Z[:, :, s] = X[:, :, s] @ self.U_site[s]
            Z_prime[:, :, s] = X_prime[:, :, s] @ self.U_prime_site[s]


        ZC = self._combine_Z_and_Z_prime(Z, Z_prime)
        Q_syn = self._destandardize_flows(ZC)

        if self.params['generate_using_log_flow']:
            Q_syn = np.exp(Q_syn)

        Q_flat = self._reshape_output(Q_syn)

        if as_array:
            return Q_flat
        else:
            if synthetic_index is None:
                synthetic_index = self._get_synthetic_index(n_years)
            return pd.DataFrame(Q_flat, columns=self.site_names, index=synthetic_index)

    def generate(self, n_realizations=1, n_years=None, as_array=True):
        if not self.fitted:
            raise RuntimeError("Call preprocessing() and fit() before generate().")
        if n_years is None:
            n_years = self.n_historic_years
        reals = [self.generate_single_series(n_years, as_array=as_array) for _ in range(n_realizations)]
        return np.stack(reals, axis=0) if as_array else {i: r for i, r in enumerate(reals)}


# import numpy as np
# import pandas as pd
# import warnings

# from sglib.core.base import Generator

# class KirschGenerator(Generator):
#     """
#     Kirsch-Nowak Synthetic Streamflow Generator
    
#     This class implements the Kirsch-Nowak algorithm for generating synthetic streamflow
#     data. The method uses bootstrapping combined with a Cholesky decomposition approach
#     to preserve both spatial correlation (between sites) and temporal correlation
#     (within and between years).
    
#     The implementation follows the methodology described in Kirsch et al. 2013:
#     1. Standardize historic flows
#     2. Create both normal and shifted (Y') data matrices 
#     3. Bootstrap monthly flows while preserving cross-site correlations
#     4. Apply dual Cholesky decompositions to preserve temporal correlations
#     5. Combine synthetic years using a mid-year split approach
#     6. De-standardize to get final synthetic flows
#     """

#     def __init__(self, 
#                  Q: pd.DataFrame, 
#                  generate_using_log_flow=True,
#                  matrix_repair_method='spectral',
#                  debug=False):
#         """
#         Initialize the Kirsch-Nowak Generator
        
#         Parameters
#         ----------
#         Q : pd.DataFrame
#             DataFrame containing historic streamflow data
#             Index must be a DatetimeIndex
#             Columns are different sites/stations

#         generate_using_log_flow: bool (default: True)
#             Whether to transform flows using logarithm
#         matrix_repair_method: str (default: 'spectral')
#             Method to repair non-positive definite matrices
#         debug: bool (default: False)
#             Whether to log debug information
#         """
#         if not isinstance(Q, pd.DataFrame):
#             raise TypeError("Input must be a pandas DataFrame.")
#         if not isinstance(Q.index, pd.DatetimeIndex):
#             raise TypeError("Input index must be a pd.DatetimeIndex.")

#         self.Q = Q.copy()
#         self.params = {
#             'generate_using_log_flow': generate_using_log_flow,
#             'matrix_repair_method': matrix_repair_method,
#             'debug': debug,
#         }

#         self.fitted = False
        
#         ## Use historic Q to get some specifications
#         self.n_historic_years = Q.index.year.nunique()
#         self.n_sites = Q.shape[1]  # Number of sites
#         self.site_names = Q.columns.tolist()
#         self.n_months = 12  # Monthly data (12 months per year)
    
#     def _get_synthetic_index(self, n_years):
#         """
#         Generate a monthly datetime index for synthetic data.
#         """
#         # synthetic index will start after the last year of historic data
#         synthetic_index = pd.date_range(start=f"{self.Q.index.year.max() + 1}-01-01",
#                                        periods=n_years * self.n_months, 
#                                        freq='MS')
#         return synthetic_index

#     def preprocessing(self, timestep='monthly'):
#         """
#         Preprocess the input data to prepare for model fitting
        
#         Parameters
#         ----------
#         timestep : str, default='monthly'
#             Time aggregation level, currently only 'monthly' is supported
#         """
#         # Group the daily data into monthly totals
#         if timestep == 'monthly':
#             if self.params['debug']:
#                 print("Aggregating data to monthly timestep...")
            
#             # Group by year and month to get monthly totals
#             monthly = self.Q.groupby([self.Q.index.year, self.Q.index.month]).sum()
#             monthly.index = pd.MultiIndex.from_tuples(monthly.index, names=['year', 'month'])
            
#             # Reshape to have months as columns for each site
#             self.Qm = monthly
            
#             if self.params['debug']:
#                 print(f"Monthly data shape: {self.Qm.shape}")
#         else:
#             raise NotImplementedError("Currently only monthly timestep is supported.")

#         # Apply logarithmic transformation if specified
#         if self.params['generate_using_log_flow']:
#             if self.params['debug']:
#                 print("Applying logarithmic transformation...")
            
#             # Ensure non-negative values before taking log
#             self.Qm = np.log(self.Qm.clip(lower=1e-6))

#     def fit(self):
#         """
#         Fit the Kirsch-Nowak model to the preprocessed data
        
#         This method:
#         1. Computes monthly means and standard deviations
#         2. Standardizes historic flows
#         3. Creates both normal (Y) and shifted (Y') matrices
#         4. Builds correlation matrices for both Y and Y'
#         5. Computes Cholesky decompositions for both
#         """
#         if self.params['debug']:
#             print("Fitting Kirsch-Nowak model...")
        
#         # Compute monthly means and stds for standardization
#         self.mean_month = self.Qm.groupby(level='month').mean()
#         self.std_month = self.Qm.groupby(level='month').std()
        
#         if self.params['debug']:
#             print(f"Mean monthly flows shape: {self.mean_month.shape}")
        
#         # Standardize historic flows using monthly statistics (whitening)
#         Z_h = pd.DataFrame(index=self.Qm.index, columns=self.Qm.columns)
        
#         for month in range(1, 13):
#             # Get data for this month across all years
#             month_data = self.Qm.xs(month, level='month')
            
#             # Standardize using month-specific mean and std
#             for col in month_data.columns:
#                 Z_h.loc[(slice(None), month), col] = (
#                     (self.Qm.loc[(slice(None), month), col] - self.mean_month.loc[month, col]) / 
#                     self.std_month.loc[month, col]
#                 )
        
#         # Remove any rows with NaN values
#         self.Z_h = Z_h.dropna()
        
#         # Store historic years for bootstrap sampling
#         self.historic_years = self.Z_h.index.get_level_values('year').unique()
#         self.n_historic_years = len(self.historic_years)
        
#         if self.params['debug']:
#             print(f"Historic years: {self.n_historic_years}, Sites: {self.n_sites}")
        
#         # Create both Y and Y' matrices as described in the paper
#         self.Y, self.Y_prime = self._create_Y_and_Y_prime_matrices()
        
#         # Build correlation matrices for both Y and Y'
#         self.corr_Y = np.corrcoef(self.Y, rowvar=False)
#         self.corr_Y_prime = np.corrcoef(self.Y_prime, rowvar=False)
        
#         if self.params['debug']:
#             print(f"Y correlation matrix shape: {self.corr_Y.shape}")
#             print(f"Y' correlation matrix shape: {self.corr_Y_prime.shape}")
        
#         # Decompose correlation matrices using Cholesky decomposition
#         self.U = self._repair_and_cholesky(self.corr_Y)
#         self.U_prime = self._repair_and_cholesky(self.corr_Y_prime)
        
#         # Store the month-to-row mapping for Y and Y'
#         self.year_to_row = {year: idx for idx, year in enumerate(self.historic_years)}

        
#         self.fitted = True
        
#         if self.params['debug']:
#             print("Model fitting complete.")

#     def _create_Y_and_Y_prime_matrices(self):
#         """
#         Create both Y and Y' matrices as described in the paper.
        
#         Y contains the normal standardized data arranged as:
#         - Each row represents one year
#         - Each column represents one site-month combination
        
#         Y' contains the same data but shifted by 6 months (half year):
#         - First column contains data from month 7 (July)
#         - 12th column contains data from month 6 (June) of following year
#         - Has one fewer row than Y due to the shifting
        
#         Returns
#         -------
#         Y : ndarray
#             Normal data matrix (n_years, n_sites * n_months)
#         Y_prime : ndarray  
#             Shifted data matrix (n_years-1, n_sites * n_months)
#         """
#         if self.params['debug']:
#             print("Creating Y and Y' matrices...")
        
#         # Create Y matrix - flatten standardized data by year
#         Y_data = []
#         for year in self.historic_years:
#             year_data = []
#             for month in range(1, 13):
#                 for site in range(self.n_sites):
#                     site_col = self.Z_h.columns[site]
#                     try:
#                         # Get standardized value for this year-month-site combination
#                         value = self.Z_h.loc[(year, month), site_col]
#                         year_data.append(value)
#                     except KeyError:
#                         # Handle missing months
#                         if self.params['debug']:
#                             print(f"Missing data for year {year}, month {month}, site {site_col}")
#                         year_data.append(0.0)
#             Y_data.append(year_data)
        
#         Y = np.array(Y_data)
        
#         # Create Y' matrix - same as Y but shifted by 6 months
#         # Remove first and last 6 months, then reshape
#         half_months = 6  # 6 months for half year
        
#         Y_prime_data = []
#         for year_idx in range(len(self.historic_years) - 1):  # One fewer year due to shifting
#             year_data = []
            
#             # First half comes from months 7-12 of current year
#             current_year = self.historic_years[year_idx]
#             for month in range(7, 13):  # months 7-12 (July-December)
#                 for site in range(self.n_sites):
#                     site_col = self.Z_h.columns[site]
#                     try:
#                         value = self.Z_h.loc[(current_year, month), site_col]
#                         year_data.append(value)
#                     except KeyError:
#                         year_data.append(0.0)
            
#             # Second half comes from months 1-6 of following year
#             next_year = self.historic_years[year_idx + 1]
#             for month in range(1, 7):  # months 1-6 (January-June)
#                 for site in range(self.n_sites):
#                     site_col = self.Z_h.columns[site]
#                     try:
#                         value = self.Z_h.loc[(next_year, month), site_col]
#                         year_data.append(value)
#                     except KeyError:
#                         year_data.append(0.0)
            
#             Y_prime_data.append(year_data)
        
#         Y_prime = np.array(Y_prime_data)
        
#         if self.params['debug']:
#             print(f"Y matrix shape: {Y.shape}")
#             print(f"Y' matrix shape: {Y_prime.shape}")
        
#         return Y, Y_prime

#     def _repair_and_cholesky(self, corr):
#         """
#         Repair a correlation matrix if not positive definite and compute Cholesky decomposition
        
#         Parameters
#         ----------
#         corr : ndarray
#             Correlation matrix
            
#         Returns
#         -------
#         ndarray
#             Upper triangular Cholesky factor U where corr = U^T * U
#         """
#         try:
#             # Try direct Cholesky decomposition
#             L = np.linalg.cholesky(corr)
#             return L.T  # Return upper triangular U
#         except np.linalg.LinAlgError:
#             if self.params['debug']:
#                 print("Matrix not positive definite, repairing...")
            
#             # Repair the matrix using eigenvalue adjustment
#             non_psd = True
#             iter_count = 0
#             repaired_corr = corr.copy()
            
#             while non_psd and iter_count < 50:
#                 # Compute eigenvalues and eigenvectors
#                 evals, evecs = np.linalg.eigh(repaired_corr)
                
#                 # Set negative eigenvalues to small positive value
#                 evals[evals < 0] = 1e-8
                
#                 # Reconstruct matrix
#                 repaired_corr = evecs @ np.diag(evals) @ evecs.T
                
#                 # Enforce diagonal of 1s (correlation matrix property)
#                 np.fill_diagonal(repaired_corr, 1.0)
                
#                 iter_count += 1
                
#                 # Check if the matrix is now positive definite
#                 try:
#                     np.linalg.cholesky(repaired_corr)
#                     non_psd = False
#                 except np.linalg.LinAlgError:
#                     non_psd = True
            
#             if non_psd:
#                 warnings.warn("Matrix could not be repaired to be positive definite.")
#                 raise ValueError("Matrix is not positive definite after 50 iterations.")
#             else:
#                 if self.params['debug']:
#                     print(f"Matrix repaired successfully after {iter_count} iterations.")
                
#                 # Return the upper triangular Cholesky factor
#                 return np.linalg.cholesky(repaired_corr).T

#     def _get_bootstrap_indices(self, n_years):
#         """
#         Generate bootstrap indices matrix M of shape (n_years, 12),
#         each value is the index of a historic year to sample for that month.
#         """
#         if self.params['debug']:
#             print(f"Generating bootstrap indices for {n_years} years...")
            
#         year_indices = np.arange(self.n_historic_years)
#         M = np.random.choice(year_indices, size=(n_years, self.n_months), replace=True)
#         return M


#     def generate_single_series(self, n_years, M=None,
#                                as_array=True,
#                                synthetic_index=None):
#         """
#         Generate a single synthetic streamflow series
        
#         Parameters
#         ----------
#         n_years : int
#             Number of years to generate
#         M : ndarray, optional
#             Bootstrap indices matrix of shape (n_years, n_months)
#             If None, will be randomly generated
#         as_array : bool, default=True
#             If True, return as numpy array of shape (n_years*n_months, n_sites)
#             If False, return as DataFrame with DatetimeIndex and site columns
            
#         Returns
#         -------
#         ndarray or DataFrame
#             Synthetic flows
#         """
#         # Add 1 to n_years to account for the mid-year split
#         # which will lose 1 year in the combination process
#         n_years_with_buffer = n_years + 1
        
#         if not self.fitted:
#             raise RuntimeError("Call preprocessing() and fit() before generate().")
        
#         if self.params['debug']:
#             print(f"Generating synthetic series for {n_years} years...")
        
#         # Generate bootstrap indices if not provided
#         if M is None:
#             M = self._get_bootstrap_indices(n_years_with_buffer)
#             M_prime = M[:self.Y_prime.shape[0], :]
#         else:
#             # Validate provided indices
#             if not isinstance(M, np.ndarray):
#                 M = np.array(M)
#             if M.shape != (n_years_with_buffer, self.n_months):
#                 raise ValueError(f"M must have shape ({n_years_with_buffer}, {self.n_months})")
        
#         # Total number of variables (site-month combinations)
#         n_vars = self.n_sites * self.n_months
        
#         # Create bootstrap matrices X and X' using the same M
#         X = self._create_bootstrap_matrix(M, use_Y_prime=False)
#         X_prime = self._create_bootstrap_matrix(M_prime, use_Y_prime=True)
        
#         # Apply correlation structures using Cholesky decomposition
#         Z = X @ self.U
#         Z_prime = X_prime @ self.U_prime
        
#         # Combine Z and Z' using mid-year split approach
#         Z_combined = self._combine_Z_and_Z_prime(Z, Z_prime)
        
#         # De-standardize the combined flows
#         Q_syn = self._destandardize_flows(Z_combined)
        
#         # Apply exponential if using log flows
#         if self.params['generate_using_log_flow']:
#             Q_syn = np.exp(Q_syn)
        
#         # Reshape to output format (n_years * n_months, n_sites)
#         Q_syn_reshaped = self._reshape_output(Q_syn, n_years)
        
#         if as_array:
#             return Q_syn_reshaped
#         else:
#             # Return as DataFrame with DatetimeIndex
#             if synthetic_index is None:
#                 synthetic_index = self._get_synthetic_index(n_years)
            
#             assert len(synthetic_index) == Q_syn_reshaped.shape[0], \
#                 f"Synthetic index length mismatch. Expected: {Q_syn_reshaped.shape[0]}, Got: {len(synthetic_index)}"
            
#             Q_syn_df = pd.DataFrame(Q_syn_reshaped, 
#                                    columns=self.site_names, 
#                                    index=synthetic_index)
            
#             return Q_syn_df

#     def _create_bootstrap_matrix(self, M, 
#                                  use_Y_prime=False):
#         """
#         Create bootstrap matrix X or X' from bootstrap indices M

#         Parameters
#         ----------
#         M : ndarray
#             Bootstrap indices matrix of shape (n_years, 12).
#             Each row contains indices of historic years to sample from,
#             one per month.
#         use_Y_prime : bool
#             If True, use Y' matrix (shifted), otherwise use Y.

#         Returns
#         -------
#         ndarray
#             Bootstrap matrix X or X' of shape (n_years, n_sites * 12)
#         """
#         n_years, n_months = M.shape
#         assert n_months == self.n_months, "M must have 12 columns for 12 months."

#         # Select the source matrix
#         source = self.Y_prime if use_Y_prime else self.Y
#         max_idx = source.shape[0] - 1  # max valid row index

#         # Output matrix
#         X = np.zeros((n_years, self.n_sites * self.n_months))

#         for i in range(n_years):
#             row_data = []
#             for m in range(self.n_months):
#                 h_idx = min(M[i, m], max_idx)  # bound index
#                 # Get the slice corresponding to month `m` for all sites
#                 start = m * self.n_sites
#                 end = (m + 1) * self.n_sites
#                 row_data.extend(source[h_idx, start:end])
#             X[i, :] = row_data

#         return X


#     def _combine_Z_and_Z_prime(self, Z, Z_prime):
#         """
#         Combine Z and Z' matrices using mid-year split approach
        
#         According to the paper:
#         "The 6-month segments that comprise ZC originate from the right
#         halves of Z and Z', which are correlated to the 6-month segments
#         on the left-hand sides of Z and Z'."
        
#         Parameters
#         ----------
#         Z : ndarray
#             Correlated matrix from normal data
#         Z_prime : ndarray
#             Correlated matrix from shifted data
            
#         Returns
#         -------
#         ndarray
#             Combined matrix ZC
#         """
#         n_years = min(Z.shape[0], Z_prime.shape[0]) - 1  # Lose one year in combination
#         n_vars = Z.shape[1]
#         half_vars = n_vars // 2  # 6 months * n_sites
        
#         Z_combined = np.zeros((n_years, n_vars))
        
#         for i in range(n_years):
#             # First half of synthetic year i comes from second half of Z'[i]
#             Z_combined[i, :half_vars] = Z_prime[i, half_vars:]
            
#             # Second half of synthetic year i comes from first half of Z[i+1]
#             Z_combined[i, half_vars:] = Z[i+1, :half_vars]
        
#         return Z_combined

#     def _destandardize_flows(self, Z_combined):
#         """
#         De-standardize the combined flows using month-specific means and stds
        
#         Parameters
#         ----------
#         Z_combined : ndarray
#             Combined standardized flows
            
#         Returns
#         -------
#         ndarray
#             De-standardized flows
#         """
#         Q_syn = np.zeros_like(Z_combined)
        
#         # Apply the appropriate mean and std for each site-month combination
#         for month in range(1, 13):
#             for site in range(self.n_sites):
#                 # Calculate indices in the flattened array
#                 flat_idx = (month - 1) * self.n_sites + site
                
#                 # Get column name for this site
#                 site_col = self.site_names[site]
                
#                 # Apply de-standardization
#                 mean_val = self.mean_month.loc[month, site_col]
#                 std_val = self.std_month.loc[month, site_col]
                
#                 Q_syn[:, flat_idx] = Z_combined[:, flat_idx] * std_val + mean_val
        
#         return Q_syn

#     def _reshape_output(self, Q_syn, n_years):
#         """
#         Reshape synthetic flows to output format (n_years * n_months, n_sites)
        
#         Parameters
#         ----------
#         Q_syn : ndarray
#             Synthetic flows in flattened format
#         n_years : int
#             Number of years
            
#         Returns
#         -------
#         ndarray
#             Reshaped flows (n_years * n_months, n_sites)
#         """
#         # Reshape to output format (n_years * n_months, n_sites)
#         Q_syn_reshaped = np.zeros((n_years * self.n_months, self.n_sites))
        
#         for site in range(self.n_sites):
#             for month in range(self.n_months):
#                 # Extract values for this site and month across all years
#                 flat_idx = month * self.n_sites + site
                
#                 # For each year, place the value in the correct position
#                 for year in range(n_years):
#                     if year < Q_syn.shape[0]:  # Ensure we don't exceed available data
#                         time_idx = year * self.n_months + month
#                         Q_syn_reshaped[time_idx, site] = Q_syn[year, flat_idx]
        
#         return Q_syn_reshaped

#     def generate(self, n_realizations=1, 
#                  n_years=None,
#                  as_array=True):
#         """
#         Generate multiple realizations of synthetic streamflow
        
#         Parameters
#         ----------
#         n_realizations : int, default=1
#             Number of independent synthetic series to generate
#         n_years : int, optional
#             Number of years in each realization (default: same as historic)
#         as_array : bool, default=True
#             If True, return as numpy array
#             If False, return as dict of DataFrames
            
#         Returns
#         -------
#         ndarray or dict
#             Generated synthetic flows
#         """
#         if not self.fitted:
#             raise RuntimeError("Call preprocessing() and fit() before generate().")
        
#         if n_years is None:
#             n_years = self.n_historic_years
        
#         if self.params['debug']:
#             print(f"Generating {n_realizations} realizations of {n_years} years each...")
        
#         # Generate the specified number of realizations
#         reals = []
#         for i in range(n_realizations):
#             if self.params['debug']:
#                 print(f"Generating realization {i+1}/{n_realizations}...")

#             # Generate a single synthetic series
#             syn = self.generate_single_series(n_years, as_array=as_array)
#             reals.append(syn)
        
#         if as_array:
#             # Stack the realizations along the first dimension
#             return np.stack(reals, axis=0)
#         else:
#             # Return a dict of dataframes for each realization
#             return {i: syn for i, syn in enumerate(reals)}
