import numpy as np
import pandas as pd
import warnings

from sglib.core.base import Generator

class KirschGenerator(Generator):
    """
    Kirsch-Nowak Synthetic Streamflow Generator
    
    This class implements the Kirsch-Nowak algorithm for generating synthetic streamflow
    data. The method uses bootstrapping combined with a Cholesky decomposition approach
    to preserve both spatial correlation (between sites) and temporal correlation
    (within and between years).
    
    The implementation follows the methodology described in Kirsch et al. 2013:
    1. Standardize historic flows
    2. Create both normal and shifted (Y') data matrices 
    3. Bootstrap monthly flows while preserving cross-site correlations
    4. Apply dual Cholesky decompositions to preserve temporal correlations
    5. Combine synthetic years using a mid-year split approach
    6. De-standardize to get final synthetic flows
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
        self.n_months = 12  # Monthly data (12 months per year)
    
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
        3. Creates both normal (Y) and shifted (Y') matrices
        4. Builds correlation matrices for both Y and Y'
        5. Computes Cholesky decompositions for both
        """
        if self.params['debug']:
            print("Fitting Kirsch-Nowak model...")
        
        # Compute monthly means and stds for standardization
        self.mean_month = self.Qm.groupby(level='month').mean()
        self.std_month = self.Qm.groupby(level='month').std()
        
        if self.params['debug']:
            print(f"Mean monthly flows shape: {self.mean_month.shape}")
        
        # Standardize historic flows using monthly statistics (whitening)
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
        
        if self.params['debug']:
            print(f"Historic years: {self.n_historic_years}, Sites: {self.n_sites}")
        
        # Create both Y and Y' matrices as described in the paper
        self.Y, self.Y_prime = self._create_Y_and_Y_prime_matrices()
        
        # Build correlation matrices for both Y and Y'
        self.corr_Y = np.corrcoef(self.Y, rowvar=False)
        self.corr_Y_prime = np.corrcoef(self.Y_prime, rowvar=False)
        
        if self.params['debug']:
            print(f"Y correlation matrix shape: {self.corr_Y.shape}")
            print(f"Y' correlation matrix shape: {self.corr_Y_prime.shape}")
        
        # Decompose correlation matrices using Cholesky decomposition
        self.U = self._repair_and_cholesky(self.corr_Y)
        self.U_prime = self._repair_and_cholesky(self.corr_Y_prime)
        
        self.fitted = True
        
        if self.params['debug']:
            print("Model fitting complete.")

    def _create_Y_and_Y_prime_matrices(self):
        """
        Create both Y and Y' matrices as described in the paper.
        
        Y contains the normal standardized data arranged as:
        - Each row represents one year
        - Each column represents one site-month combination
        
        Y' contains the same data but shifted by 6 months (half year):
        - First column contains data from month 7 (July)
        - 12th column contains data from month 6 (June) of following year
        - Has one fewer row than Y due to the shifting
        
        Returns
        -------
        Y : ndarray
            Normal data matrix (n_years, n_sites * n_months)
        Y_prime : ndarray  
            Shifted data matrix (n_years-1, n_sites * n_months)
        """
        if self.params['debug']:
            print("Creating Y and Y' matrices...")
        
        # Create Y matrix - flatten standardized data by year
        Y_data = []
        for year in self.historic_years:
            year_data = []
            for month in range(1, 13):
                for site in range(self.n_sites):
                    site_col = self.Z_h.columns[site]
                    try:
                        # Get standardized value for this year-month-site combination
                        value = self.Z_h.loc[(year, month), site_col]
                        year_data.append(value)
                    except KeyError:
                        # Handle missing months
                        if self.params['debug']:
                            print(f"Missing data for year {year}, month {month}, site {site_col}")
                        year_data.append(0.0)
            Y_data.append(year_data)
        
        Y = np.array(Y_data)
        
        # Create Y' matrix - same as Y but shifted by 6 months
        # Remove first and last 6 months, then reshape
        half_months = 6  # 6 months for half year
        
        Y_prime_data = []
        for year_idx in range(len(self.historic_years) - 1):  # One fewer year due to shifting
            year_data = []
            
            # First half comes from months 7-12 of current year
            current_year = self.historic_years[year_idx]
            for month in range(7, 13):  # months 7-12 (July-December)
                for site in range(self.n_sites):
                    site_col = self.Z_h.columns[site]
                    try:
                        value = self.Z_h.loc[(current_year, month), site_col]
                        year_data.append(value)
                    except KeyError:
                        year_data.append(0.0)
            
            # Second half comes from months 1-6 of following year
            next_year = self.historic_years[year_idx + 1]
            for month in range(1, 7):  # months 1-6 (January-June)
                for site in range(self.n_sites):
                    site_col = self.Z_h.columns[site]
                    try:
                        value = self.Z_h.loc[(next_year, month), site_col]
                        year_data.append(value)
                    except KeyError:
                        year_data.append(0.0)
            
            Y_prime_data.append(year_data)
        
        Y_prime = np.array(Y_prime_data)
        
        if self.params['debug']:
            print(f"Y matrix shape: {Y.shape}")
            print(f"Y' matrix shape: {Y_prime.shape}")
        
        return Y, Y_prime

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
        ndarray or DataFrame
            Synthetic flows
        """
        # Add 1 to n_years to account for the mid-year split
        # which will lose 1 year in the combination process
        n_years_with_buffer = n_years + 1
        
        if not self.fitted:
            raise RuntimeError("Call preprocessing() and fit() before generate().")
        
        if self.params['debug']:
            print(f"Generating synthetic series for {n_years} years...")
        
        # Generate bootstrap indices if not provided
        if M is None:
            M = self._get_bootstrap_indices(n_years_with_buffer)
        else:
            # Validate provided indices
            if not isinstance(M, np.ndarray):
                M = np.array(M)
            if M.shape != (n_years_with_buffer, self.n_months):
                raise ValueError(f"M must have shape ({n_years_with_buffer}, {self.n_months})")
        
        # Total number of variables (site-month combinations)
        n_vars = self.n_sites * self.n_months
        
        # Create bootstrap matrices X and X' using the same M
        X = self._create_bootstrap_matrix(M, use_Y_prime=False)
        X_prime = self._create_bootstrap_matrix(M, use_Y_prime=True)
        
        # Apply correlation structures using Cholesky decomposition
        Z = X @ self.U
        Z_prime = X_prime @ self.U_prime
        
        # Combine Z and Z' using mid-year split approach
        Z_combined = self._combine_Z_and_Z_prime(Z, Z_prime)
        
        # De-standardize the combined flows
        Q_syn = self._destandardize_flows(Z_combined)
        
        # Apply exponential if using log flows
        if self.params['generate_using_log_flow']:
            Q_syn = np.exp(Q_syn)
        
        # Reshape to output format (n_years * n_months, n_sites)
        Q_syn_reshaped = self._reshape_output(Q_syn, n_years)
        
        if as_array:
            return Q_syn_reshaped
        else:
            # Return as DataFrame with DatetimeIndex
            if synthetic_index is None:
                synthetic_index = self._get_synthetic_index(n_years)
            
            assert len(synthetic_index) == Q_syn_reshaped.shape[0], \
                f"Synthetic index length mismatch. Expected: {Q_syn_reshaped.shape[0]}, Got: {len(synthetic_index)}"
            
            Q_syn_df = pd.DataFrame(Q_syn_reshaped, 
                                   columns=self.site_names, 
                                   index=synthetic_index)
            
            return Q_syn_df

    def _create_bootstrap_matrix(self, M, use_Y_prime=False):
        """
        Create bootstrap matrix X or X' from bootstrap indices M
        
        Parameters
        ----------
        M : ndarray
            Bootstrap indices matrix
        use_Y_prime : bool
            If True, use Y' matrix, otherwise use Y matrix
            
        Returns
        -------
        ndarray
            Bootstrap matrix X or X'
        """
        n_years, n_months = M.shape
        n_vars = self.n_sites * self.n_months
        
        # Choose source matrix
        if use_Y_prime:
            source_matrix = self.Y_prime
            # For Y', we need to adjust the year indexing since it has one fewer row
            n_years = min(n_years, source_matrix.shape[0])
        else:
            source_matrix = self.Y
        
        # Create bootstrap matrix
        X = np.zeros((n_years, n_vars))
        
        for i in range(n_years):
            year_data = []
            for j in range(n_months):
                # Get historic year index (bounded by available data)
                if use_Y_prime:
                    h_idx = min(M[i, j], source_matrix.shape[0] - 1)
                else:
                    h_idx = M[i, j]
                
                # Extract data for this month across all sites
                month_start_idx = j * self.n_sites
                month_end_idx = month_start_idx + self.n_sites
                
                try:
                    year_data.extend(source_matrix[h_idx, month_start_idx:month_end_idx])
                except IndexError:
                    # Handle missing data with zeros
                    year_data.extend([0.0] * self.n_sites)
            
            X[i, :] = year_data[:n_vars]  # Ensure correct length
        
        return X

    def _combine_Z_and_Z_prime(self, Z, Z_prime):
        """
        Combine Z and Z' matrices using mid-year split approach
        
        According to the paper:
        "The 6-month segments that comprise ZC originate from the right
        halves of Z and Z', which are correlated to the 6-month segments
        on the left-hand sides of Z and Z'."
        
        Parameters
        ----------
        Z : ndarray
            Correlated matrix from normal data
        Z_prime : ndarray
            Correlated matrix from shifted data
            
        Returns
        -------
        ndarray
            Combined matrix ZC
        """
        n_years = min(Z.shape[0], Z_prime.shape[0]) - 1  # Lose one year in combination
        n_vars = Z.shape[1]
        half_vars = n_vars // 2  # 6 months * n_sites
        
        Z_combined = np.zeros((n_years, n_vars))
        
        for i in range(n_years):
            # First half of synthetic year i comes from second half of Z'[i]
            Z_combined[i, :half_vars] = Z_prime[i, half_vars:]
            
            # Second half of synthetic year i comes from first half of Z[i+1]
            Z_combined[i, half_vars:] = Z[i+1, :half_vars]
        
        return Z_combined

    def _destandardize_flows(self, Z_combined):
        """
        De-standardize the combined flows using month-specific means and stds
        
        Parameters
        ----------
        Z_combined : ndarray
            Combined standardized flows
            
        Returns
        -------
        ndarray
            De-standardized flows
        """
        Q_syn = np.zeros_like(Z_combined)
        
        # Apply the appropriate mean and std for each site-month combination
        for month in range(1, 13):
            for site in range(self.n_sites):
                # Calculate indices in the flattened array
                flat_idx = (month - 1) * self.n_sites + site
                
                # Get column name for this site
                site_col = self.site_names[site]
                
                # Apply de-standardization
                mean_val = self.mean_month.loc[month, site_col]
                std_val = self.std_month.loc[month, site_col]
                
                Q_syn[:, flat_idx] = Z_combined[:, flat_idx] * std_val + mean_val
        
        return Q_syn

    def _reshape_output(self, Q_syn, n_years):
        """
        Reshape synthetic flows to output format (n_years * n_months, n_sites)
        
        Parameters
        ----------
        Q_syn : ndarray
            Synthetic flows in flattened format
        n_years : int
            Number of years
            
        Returns
        -------
        ndarray
            Reshaped flows (n_years * n_months, n_sites)
        """
        # Reshape to output format (n_years * n_months, n_sites)
        Q_syn_reshaped = np.zeros((n_years * self.n_months, self.n_sites))
        
        for site in range(self.n_sites):
            for month in range(self.n_months):
                # Extract values for this site and month across all years
                flat_idx = month * self.n_sites + site
                
                # For each year, place the value in the correct position
                for year in range(n_years):
                    if year < Q_syn.shape[0]:  # Ensure we don't exceed available data
                        time_idx = year * self.n_months + month
                        Q_syn_reshaped[time_idx, site] = Q_syn[year, flat_idx]
        
        return Q_syn_reshaped

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
        as_array : bool, default=True
            If True, return as numpy array
            If False, return as dict of DataFrames
            
        Returns
        -------
        ndarray or dict
            Generated synthetic flows
        """
        if not self.fitted:
            raise RuntimeError("Call preprocessing() and fit() before generate().")
        
        if n_years is None:
            n_years = self.n_historic_years
        
        if self.params['debug']:
            print(f"Generating {n_realizations} realizations of {n_years} years each...")
        
        # Generate the specified number of realizations
        reals = []
        for i in range(n_realizations):
            if self.params['debug']:
                print(f"Generating realization {i+1}/{n_realizations}...")

            # Generate a single synthetic series
            syn = self.generate_single_series(n_years, as_array=as_array)
            reals.append(syn)
        
        if as_array:
            # Stack the realizations along the first dimension
            return np.stack(reals, axis=0)
        else:
            # Return a dict of dataframes for each realization
            return {i: syn for i, syn in enumerate(reals)}
