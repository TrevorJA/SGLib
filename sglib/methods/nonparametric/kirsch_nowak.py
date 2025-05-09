import numpy as np
import pandas as pd
import warnings

from sglib.core.base import Generator

class KirschNowakGenerator(Generator):
    """
    Kirsch-Nowak Synthetic Streamflow Generator
    
    This class implements the Kirsch-Nowak algorithm for generating synthetic streamflow
    data. The method uses bootstrapping combined with a Cholesky decomposition approach
    to preserve both spatial correlation (between sites) and temporal correlation
    (within and between years).
    
    The implementation follows the methodology described in Kirsch et al. 2013:
    1. Standardize historic flows
    2. Bootstrap monthly flows while preserving cross-site correlations
    3. Apply Cholesky decompositions to preserve temporal correlations
    4. Combine synthetic years using a mid-year split approach
    5. De-standardize to get final synthetic flows
    """

    def __init__(self, Q: pd.DataFrame, **kwargs):
        """
        Initialize the Kirsch-Nowak Generator
        
        Parameters
        ----------
        Q : pd.DataFrame
            DataFrame containing historic streamflow data
            Index must be a DatetimeIndex
            Columns are different sites/stations
        kwargs : dict
            Optional parameters:
            - generate_using_log_flow: bool (default: True)
              Whether to transform flows using logarithm
            - matrix_repair_method: str (default: 'spectral')
              Method to repair non-positive definite matrices
            - debug: bool (default: False)
              Whether to print debug information
        """
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
        
        ## Use historic Q to get some specifications
        self.n_historic_years = Q.index.year.nunique()
        self.n_sites = Q.shape[1]  # Number of sites
        self.site_names = Q.columns.tolist()
        self.n_months = 12  # Monthly data
    
    def _get_synthetic_index(self, n_years):
        """
        Generate a monthly datetime index for synthetic data.
        """
        
        # synthetic index will start after the last year of historic data
        synthetic_index = pd.date_range(start=f"{self.Q.index.year.max() + 1}-01-01",
                                                  periods=n_years * self.n_months, 
                                                  freq='MS')
        return synthetic_index
                                             
        

    def preprocessing(self, timestep='monthly'):
        """
        Preprocess the input data to prepare for model fitting
        
        Parameters
        ----------
        timestep : str, default='monthly'
            Time aggregation level, currently only 'monthly' is supported
        """
        # Group the daily data into monthly totals
        if timestep == 'monthly':
            if self.params['debug']:
                print("Aggregating data to monthly timestep...")
            
            # Group by year and month to get monthly totals
            monthly = self.Q.groupby([self.Q.index.year, self.Q.index.month]).sum()
            monthly.index = pd.MultiIndex.from_tuples(monthly.index, names=['year', 'month'])
            
            # Reshape to have months as columns for each site
            self.Qm = monthly
            
            if self.params['debug']:
                print(f"Monthly data shape: {self.Qm.shape}")
        else:
            raise NotImplementedError("Currently only monthly timestep is supported.")

        # Apply logarithmic transformation if specified
        if self.params['generate_using_log_flow']:
            if self.params['debug']:
                print("Applying logarithmic transformation...")
            
            # Ensure non-negative values before taking log
            self.Qm = np.log(self.Qm.clip(lower=1e-6))

    def fit(self):
        """
        Fit the Kirsch-Nowak model to the preprocessed data
        
        This method:
        1. Computes monthly means and standard deviations
        2. Standardizes historic flows
        3. Builds correlation matrices
        4. Computes Cholesky decompositions
        """
        if self.params['debug']:
            print("Fitting Kirsch-Nowak model...")
        
        # Compute monthly means and stds for standardization
        self.mean_month = self.Qm.groupby(level='month').mean()
        self.std_month = self.Qm.groupby(level='month').std()
        
        # Standardize historic flows using monthly statistics
        Z_h = pd.DataFrame(index=self.Qm.index, columns=self.Qm.columns)
        
        for month in range(1, 13):
            # Get data for this month across all years
            month_data = self.Qm.xs(month, level='month')
            
            # Standardize using month-specific mean and std
            for col in month_data.columns:
                Z_h.loc[(slice(None), month), col] = (
                    (self.Qm.loc[(slice(None), month), col] - self.mean_month.loc[month, col]) / 
                    self.std_month.loc[month, col]
                )
        
        # Remove any rows with NaN values
        self.Z_h = Z_h.dropna()
        
        # Store historic years for bootstrap sampling
        self.historic_years = self.Z_h.index.get_level_values('year').unique()
        self.n_historic_years = len(self.historic_years)
        self.n_sites = self.Q.shape[1]  # Number of sites
        self.n_months = 12
        
        if self.params['debug']:
            print(f"Historic years: {self.n_historic_years}, Sites: {self.n_sites}")
        
        # Build intraannual correlation matrix
        # This matrix captures correlation between all site-month combinations
        flat_data = []
        
        # Flatten the data to compute the full correlation matrix
        for year in self.historic_years:
            year_data = []
            for month in range(1, 13):
                try:
                    # Get standardized values for this year-month combination
                    month_data = self.Z_h.loc[(year, month)].values
                    year_data.extend(month_data)
                except KeyError:
                    # Handle missing months by using zeros
                    if self.params['debug']:
                        print(f"Missing data for year {year}, month {month}")
                    year_data.extend([0] * self.n_sites)
            
            flat_data.append(year_data)
        
        # Convert to numpy array
        flat_data = np.array(flat_data)
        
        # Compute correlation matrix
        self.corr_intra = np.corrcoef(flat_data, rowvar=False)
        
        # Build interannual correlation matrix
        # This captures correlation between consecutive years
        self.corr_inter = self._build_interyear_corr(flat_data)
        
        # Decompose correlation matrices using Cholesky decomposition
        self.U = self._repair_and_cholesky(self.corr_intra)
        self.U_prime = self._repair_and_cholesky(self.corr_inter)
        
        self.fitted = True
        
        if self.params['debug']:
            print("Model fitting complete.")

    def _build_interyear_corr(self, data):
        """
        Build the inter-year correlation matrix
        
        Parameters
        ----------
        data : ndarray
            Flattened yearly data of shape (n_years, n_sites * n_months)
            
        Returns
        -------
        ndarray
            Correlation matrix between consecutive years
        """
        if self.params['debug']:
            print("Building inter-year correlation matrix...")
        
        # Create lagged matrices for correlation calculation
        Z_current = data[:-1, :]  # Year t (all but last year)
        Z_next = data[1:, :]      # Year t+1 (all but first year)
        
        # Calculate correlation between consecutive years
        n_vars = data.shape[1]
        corr_full = np.corrcoef(Z_current.T, Z_next.T)
        
        # Extract the cross-correlation block (upper-right quadrant)
        corr = corr_full[:n_vars, n_vars:]
        
        return corr

    def _repair_and_cholesky(self, corr):
        """
        Repair a correlation matrix if not positive definite and compute Cholesky decomposition
        
        Parameters
        ----------
        corr : ndarray
            Correlation matrix
            
        Returns
        -------
        ndarray
            Upper triangular Cholesky factor U where corr = U^T * U
        """
        try:
            # Try direct Cholesky decomposition
            L = np.linalg.cholesky(corr)
            return L.T  # Return upper triangular U
        except np.linalg.LinAlgError:
            if self.params['debug']:
                print("Matrix not positive definite, repairing...")
            
            # Repair the matrix using eigenvalue adjustment
            non_psd = True
            iter_count = 0
            repaired_corr = corr.copy()
            
            while non_psd and iter_count < 50:
                # Compute eigenvalues and eigenvectors
                evals, evecs = np.linalg.eigh(repaired_corr)
                
                # Set negative eigenvalues to small positive value
                evals[evals < 0] = 1e-8
                
                # Reconstruct matrix
                repaired_corr = evecs @ np.diag(evals) @ evecs.T
                
                # Enforce diagonal of 1s (correlation matrix property)
                np.fill_diagonal(repaired_corr, 1.0)
                
                iter_count += 1
                
                # Check if the matrix is now positive definite
                try:
                    np.linalg.cholesky(repaired_corr)
                    non_psd = False
                except np.linalg.LinAlgError:
                    non_psd = True
            
            if non_psd:
                warnings.warn("Matrix could not be repaired to be positive definite.")
                raise ValueError("Matrix is not positive definite after 50 iterations.")
            else:
                if self.params['debug']:
                    print(f"Matrix repaired successfully after {iter_count} iterations.")
                
                # Return the upper triangular Cholesky factor
                return np.linalg.cholesky(repaired_corr).T

    def _get_bootstrap_indices(self, n_years):
        """
        Generate bootstrap indices for monthly sampling
        
        Parameters
        ----------
        n_years : int
            Number of years to generate
            
        Returns
        -------
        ndarray
            Bootstrap indices matrix of shape (n_years, n_months)
            Each element M[i,j] is the index of the historic year to use
            for synthetic year i, month j
        """
        if self.params['debug']:
            print(f"Generating bootstrap indices for {n_years} years...")
        
        # Create a matrix for bootstrap indices
        # Shape: (n_synthetic_years, n_months)
        M = np.zeros((n_years, self.n_months), dtype=int)
        
        # Random year indices to sample from
        year_indices = np.arange(self.n_historic_years)
        
        # For each month in each synthetic year, randomly sample a historic year index
        for i in range(n_years):
            for j in range(self.n_months):
                # Sample with replacement
                M[i, j] = np.random.choice(year_indices)
        
        return M

    def generate_single_series(self, n_years, M=None,
                               as_array=True,
                               synthetic_index=None):
        """
        Generate a single synthetic streamflow series
        
        Parameters
        ----------
        n_years : int
            Number of years to generate
        M : ndarray, optional
            Bootstrap indices matrix of shape (n_years, n_months)
            If None, will be randomly generated
        as_array : bool, default=True
            If True, return as numpy array of shape (n_years*n_months, n_sites)
            If False, return as DataFrame with DatetimeIndex and site columns
            
        Returns
        -------
        ndarray
            Synthetic flows of shape (n_sites, (n_years-1) * n_months)
        """
        # Add 1 to n_years to account for the mid-year split
        # which will cut 1 year later on
        n_years += 1
        
        if not self.fitted:
            raise RuntimeError("Call preprocessing() and fit() before generate().")
        
        if self.params['debug']:
            print(f"Generating synthetic series for {n_years} years...")
        
        # Generate bootstrap indices if not provided
        if M is None:
            M = self._get_bootstrap_indices(n_years)
        else:
            # Validate provided indices
            if not isinstance(M, np.ndarray):
                M = np.array(M)
            if M.shape != (n_years, self.n_months):
                raise ValueError(f"M must have shape ({n_years}, {self.n_months})")
        
        # Total number of variables (site-month combinations)
        n_vars = self.n_sites * self.n_months
        
        # Create bootstrap matrix C
        # Shape: (n_years, n_months * n_sites)
        C = np.zeros((n_years, n_vars))
        
        # Populate C with bootstrapped standardized flows
        for i in range(n_years):
            year_data = []
            
            for j in range(self.n_months):
                # Get historic year index
                h_idx = M[i, j]
                h_year = self.historic_years[h_idx]
                month = j + 1  # Convert to 1-indexed month
                
                try:
                    # Extract standardized values for this historic year-month
                    month_data = self.Z_h.loc[(h_year, month)].values
                    year_data.extend(month_data)
                except KeyError:
                    # Handle missing data
                    if self.params['debug']:
                        print(f"Missing data for year {h_year}, month {month}, using zeros")
                    year_data.extend([0] * self.n_sites)
            
            # Store the complete year data
            C[i, :] = year_data
        
        # Apply intraannual correlation using Cholesky decomposition
        # Z has shape (n_years, n_vars)
        Z = C @ self.U
        
        # For interannual correlation, use mid-year split as described in the documentation
        half_year = self.n_months // 2  # 6 for monthly data
        half_vars = n_vars // 2
        
        # Initialize the final synthetic series
        # We lose one year due to the combining process
        Z_combined = np.zeros((n_years - 1, n_vars))
        
        # Implement the mid-year split approach:
        # First half of synthetic year i comes from second half of year i-1
        # Second half of synthetic year i comes from first half of year i
        for i in range(n_years - 1):
            # Second half variables of previous year for first half of current year
            Z_combined[i, :half_vars] = Z[i, half_vars:]
            
            # First half variables of current year for second half of current year
            Z_combined[i, half_vars:] = Z[i+1, :half_vars]
        
        # De-standardize the combined flows
        # Initialize the output array
        Q_syn = np.zeros_like(Z_combined)
        
        # Apply the appropriate mean and std for each site-month combination
        for month in range(1, self.n_months + 1):
            for site in range(self.n_sites):
                # Calculate indices in the flattened array
                flat_idx = (month - 1) * self.n_sites + site
                
                # Get column name for this site
                site_col = self.Qm.columns[site]
                
                # Apply de-standardization
                mean_val = self.mean_month.loc[month, site_col]
                std_val = self.std_month.loc[month, site_col]
                
                Q_syn[:, flat_idx] = Z_combined[:, flat_idx] * std_val + mean_val
        
        # Apply exponential if using log flows
        if self.params['generate_using_log_flow']:
            Q_syn = np.exp(Q_syn)
        
        # Reshape to output format (n_sites, n_synthetic_years * n_months)
        Q_syn_reshaped = np.zeros((self.n_sites, (n_years - 1) * self.n_months))
        
        for site in range(self.n_sites):
            for month in range(self.n_months):
                # Extract values for this site and month across all years
                flat_idx = month * self.n_sites + site
                
                # For each year, place the value in the correct position
                for year in range(n_years - 1):
                    time_idx = year * self.n_months + month
                    Q_syn_reshaped[site, time_idx] = Q_syn[year, flat_idx]
                    
        # Format as (n_years * n_months, n_sites)
        Q_syn_reshaped = Q_syn_reshaped.T
        
        if as_array:
            # If as_array is True, return as numpy array
            return Q_syn_reshaped
        else:
            # If as_array is False, return as DataFrame with DatetimeIndex
            if synthetic_index is None:
                # Generate a synthetic index if not provided
                synthetic_index = self._get_synthetic_index(n_years - 1)

            assert len(synthetic_index) == Q_syn_reshaped.shape[0], \
                f"Synthetic index length mismatch. Expected: {Q_syn_reshaped.shape[1]}, Got: {len(synthetic_index)})"


            Q_syn_df = pd.DataFrame(Q_syn_reshaped, 
                                          columns=self.Q.columns.tolist(), 
                                          index=synthetic_index)
        
            return Q_syn_df

    def generate(self, n_realizations=1, 
                 n_years=None,
                 as_array=True):
        """
        Generate multiple realizations of synthetic streamflow
        
        Parameters
        ----------
        n_realizations : int, default=1
            Number of independent synthetic series to generate
        n_years : int, optional
            Number of years in each realization (default: same as historic)
            
        Returns
        -------
        ndarray
            Shape: (n_realizations, (n_years*n_months), n_sites)
            Each realization is an independent synthetic flow sequence
        """
        if not self.fitted:
            raise RuntimeError("Call preprocessing() and fit() before generate().")
        
        if self.params['debug']:
            print(f"Generating {n_realizations} realizations of {n_years} years each...")
        
        # Make a synthetic index for the generated series
        synthetic_index = self._get_synthetic_index(n_years)
        
        # Generate the specified number of realizations
        reals = []
        for i in range(n_realizations):
            if self.params['debug']:
                print(f"Generating realization {i+1}/{n_realizations}...")

            # Generate a single synthetic series
            syn = self.generate_single_series(n_years, 
                                              synthetic_index=synthetic_index, 
                                              as_array=as_array)
            
            # Each syn has shape (n_years * n_months, n_sites)
            reals.append(syn)
        
        # Stack the realizations along the first dimension
        # ensemble has shape (n_realizations, (n_years) * n_months, n_sites)
        if as_array:
            return np.stack(reals, axis=0)
        else:
            # If as_array is False, 
            # return a dict of dataframes for each realization {real_id: DataFrame}
            reals = {i : syn for i, syn in enumerate(reals)}
            return reals


















# import numpy as np
# import pandas as pd
# import warnings

# from sglib.core.base import Generator

# class KirschNowakGenerator(Generator):

#     def __init__(self, Q: pd.DataFrame, **kwargs):
#         if not isinstance(Q, pd.DataFrame):
#             raise TypeError("Input must be a pandas DataFrame.")
#         if not isinstance(Q.index, pd.DatetimeIndex):
#             raise TypeError("Input index must be a pd.DatetimeIndex.")

#         self.Q = Q.copy()
#         self.params = {
#             'generate_using_log_flow': True,
#             'matrix_repair_method': 'spectral',
#             'debug': False,
#         }
#         self.params.update(kwargs)
#         self.fitted = False



#     def preprocessing(self, timestep='monthly'):
#         # Group the daily data into monthly totals
#         if timestep == 'monthly':
#             monthly = self.Q.groupby([self.Q.index.year, self.Q.index.month]).sum()
#             monthly.index = pd.MultiIndex.from_tuples(monthly.index, names=['year', 'month'])
#             self.Qm = monthly.unstack(level=-1)
#         else:
#             raise NotImplementedError("Currently only monthly timestep is supported.")

#         if self.params['generate_using_log_flow']:
#             self.Qm = np.log(self.Qm.clip(lower=1e-6))

#     def fit(self):
#         # Compute monthly means and stds for standardization
#         self.mean_month = self.Qm.mean(axis=0)
#         self.std_month = self.Qm.std(axis=0)

#         # Standardize historic flows
#         Z_h = (self.Qm - self.mean_month) / self.std_month
#         self.Z_h = Z_h.dropna()

#         # Build intraannual correlation matrix
#         self.corr_intra = self.Z_h.corr()

#         # Build interannual correlation matrix using correct year shifting
#         self.corr_inter = self._build_interyear_corr(self.Z_h)

#         # Decompose correlation matrices
#         self.U = self._repair_and_cholesky(self.corr_intra)
#         self.U_prime = self._repair_and_cholesky(self.corr_inter)

#         self.historic_years = self.Z_h.index.get_level_values(0).unique()
#         self.n_years = len(self.historic_years)
#         self.n_sites = self.Q.shape[1]
#         self.n_months = 12

#         self.fitted = True

#     def _get_random_bootstrap(self, n_years):
#         M = np.random.choice(self.historic_years, size=(n_years+2,), replace=True)
#         return M

#     def generate_single_series(self, n_years, M=None):
        
#         if M is None:
#             # Bootstrap years
#             M = self._get_random_bootstrap(n_years)
#         else:
#             # Check if M is a valid bootstrap year array
#             if not isinstance(M, (list, np.ndarray)):
#                 raise TypeError("M must be a list or numpy array.")
#             if len(M) != n_years + 2:
#                 raise ValueError("Length of M must be n_years + 2.")

        
#         # Create bootstrap matrix C
#         C = np.vstack([self.Z_h.loc[year].values for year in M])

#         # Apply intraannual correlation
#         Z = (C[:-1, :] @ self.U)
#         Z_prime = (C[1:, :] @ self.U_prime)

#         # Combine Z and Z_prime properly across mid-year split
#         Z_combined = self._combine_interannual(Z, Z_prime)

#         # De-standardize
#         Q_syn = (Z_combined * self.std_month.values) + self.mean_month.values

#         if self.params['generate_using_log_flow']:
#             Q_syn = np.exp(Q_syn)

#         # Reshape to (sites, full timeseries length)
#         Q_syn = Q_syn.reshape((n_years, self.n_sites, self.n_months))
#         Q_syn = Q_syn.transpose(1, 0, 2).reshape(self.n_sites, n_years * self.n_months)

#         return Q_syn

#     def generate(self, n_realizations=1, n_years=None):
#         if not self.fitted:
#             raise RuntimeError("Call preprocessing() and fit() before generate().")

#         n_years = n_years or self.n_years
#         reals = [self.generate_single_series(n_years) for _ in range(n_realizations)]

#         return np.stack(reals, axis=0)

#     ####### Helper Methods ########

#     def _check_positive_definite(self, matrix):
#         """Check if a matrix is positive definite."""
#         try:
#             np.linalg.cholesky(matrix)
#             return True
#         except np.linalg.LinAlgError:
#             return False

#     def _repair_and_cholesky(self, corr):
#         try:
#             return np.linalg.cholesky(corr).T
#         except np.linalg.LinAlgError:
#             warnings.warn("Matrix not positive definite, repairing...")
#             non_psd = True
#             iter = 0
#             while non_psd and iter < 50:
#                 evals, evecs = np.linalg.eigh(corr)
#                 evals[evals < 0] = 1e-8
#                 corr = evecs @ np.diag(evals) @ evecs.T
#                 iter += 1
#                 try:
#                     np.linalg.cholesky(corr)
#                     non_psd = False
#                 except np.linalg.LinAlgError:
#                     non_psd = True
                    
#             if non_psd:
#                 warnings.warn("Matrix could not be repaired to be positive definite.")
#                 raise ValueError("Matrix is not positive definite after 50 iterations.")
#             else:
#                 print(f"Matrix repaired successfully after {iter} iterations.")
#                 return np.linalg.cholesky(corr).T


#     def _build_interyear_corr(self, Z_h):
#         n_years = Z_h.index.get_level_values(0).nunique()
#         n_vars = Z_h.shape[1]

#         Z_mat = Z_h.values.reshape((n_years, n_vars))
#         Z_main = Z_mat[:-1, :]
#         Z_shifted = Z_mat[1:, :]

#         # Direct correlation between year t and year t+1
#         corr = np.corrcoef(Z_main.T, Z_shifted.T)[:n_vars, n_vars:]

#         return corr

#     def _combine_interannual(self, Z, Z_prime):
#         n, t = Z.shape
#         half = t // 2

#         Z_combined = np.zeros((n-1, t))

#         # First half of each synthetic year from second half of Z'
#         Z_combined[:, :half] = Z_prime[:-1, half:]

#         # Second half from first half of Z
#         Z_combined[:, half:] = Z[1:, :half]

#         return Z_combined
