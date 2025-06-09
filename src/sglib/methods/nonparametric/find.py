"""
Implements the Frequency, INtensity, and Duration (FIND) generator as descibed in Zaniolo et al. (2024).

The original (MATLAB) code is available at:
https://github.com/m-zaniolo/FIND-drought-generator


References:
    Zaniolo, M., Fletcher, S., & Mauter, M. (2024). FIND: A synthetic weather generator to control drought frequency, intensity, and duration. Environmental Modelling & Software, 172, 105927.
"""

import numpy as np
import pandas as pd
from scipy import stats
import warnings

from sglib.core.base import Generator
from typing import Optional, Dict, Any, Tuple, Union

class FIND(Generator):
    """
    FIND (Frequency, Intensity, and Duration) drought generator.
    
    Generates synthetic streamflow time series with controlled drought characteristics
    using simulated annealing optimization. Supports both single-site and multi-site
    generation with spatial correlation preservation.
    """
    
    def __init__(self):
        """Initialize FIND generator with default parameters."""
        # Drought definition parameters
        self.min_drought_duration = 3  # months
        self.min_drought_intensity = -0.6  # SSI threshold
        self.ssi_time_scale = 12  # months for SSI calculation
        self.nmonths_end_drought = 3  # consecutive positive SSI to end drought
        
        # Generation parameters
        self.nyears_generate = 100
        self.distribution = 'pearson3'  # or 'gamma'
        
        # Optimization parameters
        self.n_iterations = 600  # swaps per temperature
        self.n_temp_reductions = 15
        self.initial_temperature = 0.0001
        self.decrease_rate = 0.8
        self.initial_nmonths = 48  # initial window size for swaps
        self.tolerance = 0.02
        
        # Objective weights [intensity, duration, frequency, autocorr, non_drought_dist]
        self.weights = np.array([1, 4, 1, 2, 2])
        self.weights = self.weights / np.sum(self.weights)
        
        # Target parameters (set during fit)
        self.target_intensity = None
        self.target_duration = None
        self.target_frequency = None
        
        # Historical statistics (computed during fit)
        self.historical_stats = {}
        self.obs_data = None
        self.site_names = None
        
    def preprocessing(self, 
                      data: pd.DataFrame, 
                      **kwargs) -> pd.DataFrame:
        """
        Preprocess input data for FIND generation.
        
        Args:
            data: DataFrame with datetime index and site columns
            **kwargs: Additional preprocessing parameters
            
        Returns:
            Preprocessed DataFrame
        """
        if not isinstance(data.index, pd.DatetimeIndex):
            raise ValueError("Data must have datetime index")
            
        # Ensure complete years only
        data = data.resample('M').mean()  # Monthly data
        start_year = data.index[0].year
        end_year = data.index[-1].year
        
        # Keep only complete years
        full_years_data = data.loc[f'{start_year}-01':f'{end_year}-12']
        n_complete_years = len(full_years_data) // 12
        full_years_data = full_years_data.iloc[:n_complete_years * 12]
        
        # Check for missing values
        if full_years_data.isnull().any().any():
            warnings.warn("Missing values detected. Consider filling before generation.")
            
        return full_years_data
    
    def fit(self, data: pd.DataFrame, target_intensity: Optional[float] = None,
            target_duration: Optional[float] = None, target_frequency: Optional[int] = None,
            **kwargs):
        """
        Fit the generator to historical data and compute target statistics.
        
        Args:
            data: Historical streamflow data
            target_intensity: Desired drought intensity (if None, uses historical mean)
            target_duration: Desired drought duration (if None, uses historical mean)
            target_frequency: Desired drought frequency (if None, uses historical scaled)
            **kwargs: Additional fitting parameters
        """
        self.obs_data = self.preprocessing(data)
        self.site_names = list(self.obs_data.columns)
        
        # Calculate historical drought statistics for first site (primary site)
        primary_site = self.obs_data.iloc[:, 0].values
        durations, intensities, frequency, ssi_hist, drought_periods = self._identify_droughts(
            primary_site, primary_site
        )
        
        # Set target parameters
        nyears_hist = len(primary_site) // 12
        self.target_intensity = target_intensity or np.mean(intensities) if len(intensities) > 0 else -1.0
        self.target_duration = target_duration or np.mean(durations) if len(durations) > 0 else 6
        self.target_frequency = target_frequency or int(frequency / nyears_hist * self.nyears_generate)
        
        # Calculate historical statistics for each site
        for i, site in enumerate(self.site_names):
            site_data = self.obs_data.iloc[:, i].values
            
            # Calculate autocorrelation
            autocorr = self._calculate_autocorrelation(site_data, 12)
            
            # Calculate non-drought distribution percentiles
            _, _, _, ssi, drought_periods = self._identify_droughts(primary_site, site_data)
            non_drought_mask = np.ones(len(ssi), dtype=bool)
            for start, end in drought_periods:
                non_drought_mask[start:end+1] = False
            
            non_drought_ssi = ssi[non_drought_mask]
            non_drought_percentiles = np.percentile(non_drought_ssi, [25, 50, 75]) if len(non_drought_ssi) > 0 else [0, 0, 1]
            
            self.historical_stats[site] = {
                'autocorr': autocorr,
                'non_drought_percentiles': non_drought_percentiles,
                'monthly_stats': self._calculate_monthly_stats(site_data)
            }
        
        # For multi-site, calculate cross-correlation
        if len(self.site_names) > 1:
            self._calculate_cross_correlations()
    
    def generate(self, n_scenarios: int = 1, sites: Optional[list] = None, **kwargs) -> pd.DataFrame:
        """
        Generate synthetic streamflow scenarios.
        
        Args:
            n_scenarios: Number of scenarios to generate
            sites: List of sites to generate (if None, generates all fitted sites)
            **kwargs: Additional generation parameters
            
        Returns:
            DataFrame with synthetic scenarios
        """
        if self.obs_data is None:
            raise ValueError("Must call fit() before generate()")
            
        sites = sites or self.site_names
        
        # Generate scenarios for primary site first
        primary_scenarios = []
        for scenario in range(n_scenarios):
            synthetic = self._generate_single_site_scenario(sites[0])
            primary_scenarios.append(synthetic)
        
        results = {sites[0]: primary_scenarios}
        
        # Generate correlated scenarios for additional sites
        for site in sites[1:]:
            site_scenarios = []
            for scenario in range(n_scenarios):
                primary_ssi = self._calculate_ssi(
                    self.obs_data.iloc[:, 0].values, 
                    primary_scenarios[scenario]
                )
                synthetic = self._generate_matched_site_scenario(site, primary_ssi)
                site_scenarios.append(synthetic)
            results[site] = site_scenarios
        
        # Format results
        return self._format_results(results, n_scenarios)
    
    def _generate_single_site_scenario(self, site: str) -> np.ndarray:
        """Generate synthetic scenario for a single site using simulated annealing."""
        obs_data = self.obs_data[site].values
        monthly_stats = self.historical_stats[site]['monthly_stats']
        
        # Initialize synthetic time series
        synthetic = self._initialize_synthetic_series(obs_data, monthly_stats)
        
        # Ensure at least one drought exists
        while True:
            durations, _, _, _, _ = self._identify_droughts(obs_data, synthetic)
            if len(durations) > 0:
                break
            synthetic = self._initialize_synthetic_series(obs_data, monthly_stats)
        
        # Simulated annealing optimization
        current_series = synthetic.copy()
        temperature = self.initial_temperature
        nmonths = self.initial_nmonths
        
        for temp_step in range(self.n_temp_reductions):
            temperature *= self.decrease_rate
            nmonths = max(1, int(nmonths * self.decrease_rate))
            
            # Prepare cumulative distributions for this time scale
            cum_distributions = self._prepare_cumulative_distributions(obs_data, nmonths)
            
            for iteration in range(self.n_iterations):
                # Generate candidate swap
                candidate_series = self._perform_swap(
                    current_series, obs_data, cum_distributions, nmonths
                )
                
                # Calculate objectives
                obj_current = self._calculate_objectives(obs_data, current_series, site)
                obj_candidate = self._calculate_objectives(obs_data, candidate_series, site)
                
                # Accept or reject based on simulated annealing criterion
                if self._accept_swap(obj_current, obj_candidate, temperature):
                    current_series = candidate_series
                
                # Check convergence
                if obj_current < self.tolerance:
                    return current_series
        
        return current_series
    
    def _generate_matched_site_scenario(self, site: str, target_ssi: np.ndarray) -> np.ndarray:
        """Generate synthetic scenario that matches correlation with target SSI."""
        obs_data = self.obs_data[site].values
        monthly_stats = self.historical_stats[site]['monthly_stats']
        
        # Calculate target dispersion (sum of squared differences)
        primary_obs = self.obs_data.iloc[:, 0].values
        obs_ssi_primary = self._calculate_ssi(primary_obs, primary_obs)
        obs_ssi_site = self._calculate_ssi(primary_obs, obs_data)
        target_dispersion = np.sum((obs_ssi_site - obs_ssi_primary) ** 2)
        
        # Initialize and optimize similar to single site but with dispersion objective
        synthetic = self._initialize_synthetic_series(obs_data, monthly_stats)
        current_series = synthetic.copy()
        temperature = self.initial_temperature
        nmonths = self.initial_nmonths
        
        for temp_step in range(self.n_temp_reductions):
            temperature *= self.decrease_rate
            nmonths = max(1, int(nmonths * self.decrease_rate))
            
            cum_distributions = self._prepare_cumulative_distributions(obs_data, nmonths)
            
            for iteration in range(self.n_iterations):
                candidate_series = self._perform_swap(
                    current_series, obs_data, cum_distributions, nmonths
                )
                
                # Calculate dispersion objectives
                current_ssi = self._calculate_ssi(primary_obs, current_series)
                candidate_ssi = self._calculate_ssi(primary_obs, candidate_series)
                
                current_dispersion = np.sum((current_ssi - target_ssi) ** 2)
                candidate_dispersion = np.sum((candidate_ssi - target_ssi) ** 2)
                
                obj_current = abs(target_dispersion - current_dispersion)
                obj_candidate = abs(target_dispersion - candidate_dispersion)
                
                if self._accept_swap(obj_current, obj_candidate, temperature):
                    current_series = candidate_series
                
                if obj_current < self.tolerance:
                    return current_series
        
        return current_series
    
    def _identify_droughts(self, obs_ref: np.ndarray, obs_calc: np.ndarray) -> Tuple:
        """Identify drought periods using SSI threshold method."""
        ssi = self._calculate_ssi(obs_ref, obs_calc)
        
        # Find drought periods
        drought_periods = []
        intensities = []
        in_drought = False
        drought_start = 0
        positive_count = 0
        negative_count = 0
        
        for i, ssi_val in enumerate(ssi):
            if not in_drought:
                if ssi_val < 0:
                    if negative_count == 0:
                        drought_start = i
                    negative_count += 1
                    positive_count = max(0, positive_count - 1)
                    
                    if negative_count >= self.min_drought_duration:
                        in_drought = True
                        positive_count = 0
                        negative_count = 0
                else:
                    positive_count += 1
                    negative_count = max(0, negative_count - 1)
                    if positive_count > 6:
                        negative_count = 0
                        positive_count = 0
            else:
                if positive_count == 0:
                    drought_end = i
                
                if ssi_val >= 0:
                    positive_count += 1
                    if positive_count >= self.nmonths_end_drought:
                        # End drought
                        drought_periods.append((drought_start, drought_end - 1))
                        intensities.append(np.mean(ssi[drought_start:drought_end]))
                        in_drought = False
                        positive_count = 0
                        negative_count = 0
                else:
                    positive_count = max(0, positive_count - 1)
        
        # Handle drought extending to end of series
        if in_drought:
            drought_periods.append((drought_start, len(ssi) - 1))
            intensities.append(np.mean(ssi[drought_start:]))
        
        # Filter by intensity threshold
        valid_droughts = []
        valid_intensities = []
        for (start, end), intensity in zip(drought_periods, intensities):
            if intensity < self.min_drought_intensity:
                valid_droughts.append((start, end))
                valid_intensities.append(intensity)
        
        durations = [end - start + 1 for start, end in valid_droughts]
        frequency = len(valid_droughts)
        
        return durations, valid_intensities, frequency, ssi, valid_droughts
    
    def _calculate_ssi(self, obs_ref: np.ndarray, obs_calc: np.ndarray) -> np.ndarray:
        """Calculate Standardized Streamflow Index (SSI)."""
        # Reshape data for monthly analysis
        nyears_ref = len(obs_ref) // 12
        nyears_calc = len(obs_calc) // 12
        
        ref_monthly = obs_ref[:nyears_ref * 12].reshape(nyears_ref, 12)
        calc_monthly = obs_calc[:nyears_calc * 12].reshape(nyears_calc, 12)
        
        # Aggregate to specified time scale
        if self.ssi_time_scale > 1:
            ref_agg = self._rolling_sum(obs_ref, self.ssi_time_scale)
            calc_agg = self._rolling_sum(obs_calc, self.ssi_time_scale)
        else:
            ref_agg = obs_ref
            calc_agg = obs_calc
        
        # Calculate SSI for each month
        ssi = np.full(len(calc_agg), np.nan)
        
        for month in range(12):
            # Get data for this calendar month
            ref_month_data = ref_agg[month::12]
            calc_month_data = calc_agg[month::12]
            
            # Fit distribution and calculate SSI
            if self.distribution == 'pearson3':
                ssi_month = self._calculate_ssi_pearson3(ref_month_data, calc_month_data)
            else:  # gamma
                ssi_month = self._calculate_ssi_gamma(ref_month_data, calc_month_data)
            
            ssi[month::12] = ssi_month[:len(ssi[month::12])]
        
        # Trim to account for aggregation
        if self.ssi_time_scale > 1:
            ssi = ssi[self.ssi_time_scale - 1:]
        
        # Clip extreme values
        ssi = np.clip(ssi, -4, 4)
        
        return ssi
    
    def _calculate_ssi_pearson3(self, ref_data: np.ndarray, calc_data: np.ndarray) -> np.ndarray:
        """Calculate SSI using Pearson Type III distribution."""
        # Calculate L-moments
        l_moments = self._calculate_l_moments(ref_data)
        
        # Fit Pearson III parameters
        if abs(l_moments['tau3']) <= 1e-6:
            # Normal distribution case
            loc = l_moments['lambda1']
            scale = l_moments['lambda2'] * np.sqrt(np.pi)
            ssi = stats.norm.ppf(stats.norm.cdf(calc_data, loc=loc, scale=scale))
        else:
            # Pearson III distribution
            alpha = 4 / l_moments['tau3']**2
            beta = 0.5 * l_moments['lambda2'] * abs(l_moments['tau3'])
            xi = l_moments['lambda1'] - 2 * l_moments['lambda2'] / l_moments['tau3']
            
            if l_moments['tau3'] > 0:
                cdf_vals = stats.gamma.cdf((calc_data - xi) / beta, alpha)
            else:
                cdf_vals = 1 - stats.gamma.cdf((xi - calc_data) / beta, alpha)
            
            ssi = stats.norm.ppf(cdf_vals)
        
        return ssi
    
    def _calculate_ssi_gamma(self, ref_data: np.ndarray, calc_data: np.ndarray) -> np.ndarray:
        """Calculate SSI using Gamma distribution."""
        # Remove zeros for gamma fitting
        nonzero_data = ref_data[ref_data > 0]
        
        if len(nonzero_data) == 0:
            return np.zeros(len(calc_data))
        
        # Method of moments for gamma distribution
        mean_val = np.mean(nonzero_data)
        log_mean = np.mean(np.log(nonzero_data))
        u_val = np.log(mean_val) - log_mean
        
        shape = (1 + np.sqrt(1 + 4 * u_val / 3)) / (4 * u_val)
        scale = mean_val / shape
        
        # Calculate CDF and transform to standard normal
        cdf_vals = stats.gamma.cdf(calc_data, shape, scale=scale)
        ssi = stats.norm.ppf(cdf_vals)
        
        return ssi
    
    def _calculate_l_moments(self, data: np.ndarray) -> Dict[str, float]:
        """Calculate L-moments for Pearson III fitting."""
        n = len(data)
        data_sorted = np.sort(data)
        
        # Calculate probability weighted moments
        betas = []
        for r in range(3):
            weights = np.array([self._nchoosek(i, r) for i in range(n)])
            beta = np.sum(weights * data_sorted) / (n * self._nchoosek(n-1, r))
            betas.append(beta)
        
        # Convert to L-moments
        l1 = betas[0]
        l2 = 2 * betas[1] - betas[0]
        l3 = 6 * betas[2] - 6 * betas[1] + betas[0]
        
        # Calculate L-moment ratios
        tau3 = l3 / l2 if l2 != 0 else 0
        
        return {
            'lambda1': l1,
            'lambda2': l2,
            'lambda3': l3,
            'tau3': tau3
        }
    
    def _nchoosek(self, n: int, k: int) -> int:
        """Calculate binomial coefficient."""
        if k > n or k < 0:
            return 0
        if k == 0 or k == n:
            return 1
        
        k = min(k, n - k)  # Take advantage of symmetry
        result = 1
        for i in range(k):
            result = result * (n - i) // (i + 1)
        return result
    
    def _rolling_sum(self, data: np.ndarray, window: int) -> np.ndarray:
        """Calculate rolling sum with specified window size."""
        result = np.full(len(data), np.nan)
        for i in range(len(data) - window + 1):
            result[i] = np.sum(data[i:i + window])
        return result
    
    def _calculate_autocorrelation(self, data: np.ndarray, max_lag: int) -> np.ndarray:
        """Calculate autocorrelation function."""
        n = len(data)
        autocorr = np.zeros(max_lag + 1)
        
        for lag in range(max_lag + 1):
            if lag == 0:
                autocorr[lag] = 1.0
            else:
                x1 = data[:-lag]
                x2 = data[lag:]
                autocorr[lag] = np.corrcoef(x1, x2)[0, 1]
        
        return autocorr
    
    def _calculate_monthly_stats(self, data: np.ndarray) -> Dict:
        """Calculate monthly statistics for synthetic generation."""
        nyears = len(data) // 12
        monthly_data = data[:nyears * 12].reshape(nyears, 12)
        
        stats_dict = {}
        for month in range(12):
            month_values = monthly_data[:, month]
            
            # Fit distribution to monthly data
            if self.distribution == 'pearson3':
                l_moments = self._calculate_l_moments(month_values)
                stats_dict[month] = {
                    'type': 'pearson3',
                    'l_moments': l_moments,
                    'data': month_values
                }
            else:  # gamma
                nonzero = month_values[month_values > 0]
                if len(nonzero) > 0:
                    mean_val = np.mean(nonzero)
                    log_mean = np.mean(np.log(nonzero))
                    u_val = np.log(mean_val) - log_mean
                    shape = (1 + np.sqrt(1 + 4 * u_val / 3)) / (4 * u_val)
                    scale = mean_val / shape
                else:
                    shape, scale = 1, 1
                
                stats_dict[month] = {
                    'type': 'gamma',
                    'shape': shape,
                    'scale': scale,
                    'data': month_values
                }
        
        return stats_dict
    
    def _calculate_cross_correlations(self):
        """Calculate cross-correlations between sites."""
        self.cross_correlations = {}
        
        for i, site1 in enumerate(self.site_names):
            for j, site2 in enumerate(self.site_names[i+1:], i+1):
                # Calculate SSI for both sites
                obs1 = self.obs_data.iloc[:, i].values
                obs2 = self.obs_data.iloc[:, j].values
                
                ssi1 = self._calculate_ssi(obs1, obs1)
                ssi2 = self._calculate_ssi(obs1, obs2)
                
                # Calculate dispersion
                dispersion = np.sum((ssi1 - ssi2) ** 2)
                self.cross_correlations[(site1, site2)] = dispersion
    
    def _initialize_synthetic_series(self, obs_data: np.ndarray, monthly_stats: Dict) -> np.ndarray:
        """Initialize synthetic time series by sampling from monthly distributions."""
        n_months = self.nyears_generate * 12
        synthetic = np.zeros(n_months)
        
        for i in range(n_months):
            month = i % 12
            stats = monthly_stats[month]
            
            # Sample from appropriate distribution
            u = np.random.rand()
            
            if stats['type'] == 'pearson3':
                # Use inverse transform sampling with Pearson III
                month_data = stats['data']
                empirical_cdf = np.arange(1, len(month_data) + 1) / (len(month_data) + 1)
                sorted_data = np.sort(month_data)
                synthetic[i] = np.interp(u, empirical_cdf, sorted_data)
            else:  # gamma
                synthetic[i] = stats.gamma.ppf(u, stats['shape'], scale=stats['scale'])
        
        return synthetic
    
    def _prepare_cumulative_distributions(self, obs_data: np.ndarray, window_size: int) -> Dict:
        """Prepare cumulative distributions for swap operations."""
        # Calculate rolling sums
        cum_data = self._rolling_sum(obs_data, window_size)
        nyears = len(obs_data) // 12
        cum_monthly = cum_data[:nyears * 12].reshape(nyears, 12)
        
        distributions = {}
        for month in range(12):
            month_data = cum_monthly[:, month]
            month_data = month_data[~np.isnan(month_data)]
            
            if len(month_data) > 0:
                l_moments = self._calculate_l_moments(month_data)
                distributions[month] = {
                    'data': np.sort(month_data),
                    'cdf': np.arange(1, len(month_data) + 1) / (len(month_data) + 1),
                    'l_moments': l_moments
                }
            else:
                distributions[month] = {'data': np.array([0]), 'cdf': np.array([0.5])}
        
        return distributions
    
    def _perform_swap(self, current_series: np.ndarray, obs_data: np.ndarray, 
                     cum_distributions: Dict, nmonths: int) -> np.ndarray:
        """Perform a swap operation in the synthetic series."""
        candidate_series = current_series.copy()
        
        # Select random position for swap
        max_start = len(current_series) - nmonths
        start_pos = np.random.randint(0, max_start)
        month = start_pos % 12
        
        # Sample from cumulative distribution
        dist = cum_distributions[month]
        u = np.random.rand()
        cum_value = np.interp(u, dist['cdf'], dist['data'])
        
        # Disaggregate using k-NN approach
        disaggregated = self._knn_disaggregate(cum_value, nmonths, obs_data, month)
        
        # Replace values in candidate series
        candidate_series[start_pos:start_pos + nmonths] = disaggregated
        
        return candidate_series
    
    def _knn_disaggregate(self, cum_value: float, n_months: int, 
                         historical: np.ndarray, start_month: int) -> np.ndarray:
        """Disaggregate cumulative value using k-nearest neighbors."""
        # Calculate historical cumulative values for same season
        hist_cum = self._rolling_sum(historical, n_months)
        nyears = len(historical) // 12
        
        # Extract values starting in the same month
        month_indices = np.arange(start_month, len(hist_cum), 12)
        month_indices = month_indices[month_indices + n_months - 1 < len(historical)]
        
        if len(month_indices) == 0:
            # Fallback: uniform disaggregation
            return np.full(n_months, cum_value / n_months)
        
        hist_cum_month = hist_cum[month_indices]
        hist_cum_month = hist_cum_month[~np.isnan(hist_cum_month)]
        
        if len(hist_cum_month) == 0:
            return np.full(n_months, cum_value / n_months)
        
        # Find k nearest neighbors
        k = min(3, len(hist_cum_month))
        distances = np.abs(hist_cum_month - cum_value)
        nearest_indices = np.argsort(distances)[:k]
        
        # Select one neighbor with probability proportional to inverse distance
        inv_distances = 1.0 / (distances[nearest_indices] + 1e-6)
        probabilities = inv_distances / np.sum(inv_distances)
        selected_idx = np.random.choice(k, p=probabilities)
        selected_neighbor = nearest_indices[selected_idx]
        
        # Extract disaggregated pattern
        hist_start = month_indices[selected_neighbor]
        pattern = historical[hist_start:hist_start + n_months]
        
        # Scale to match cumulative value
        if np.sum(pattern) > 0:
            scaled_pattern = pattern * cum_value / np.sum(pattern)
        else:
            scaled_pattern = np.full(n_months, cum_value / n_months)
        
        return scaled_pattern
    
    def _calculate_objectives(self, obs_data: np.ndarray, synthetic: np.ndarray, site: str) -> float:
        """Calculate weighted objective function for optimization."""
        # Identify droughts in synthetic series
        durations, intensities, frequency, ssi, drought_periods = self._identify_droughts(
            obs_data, synthetic
        )
        
        # Objective 1: Intensity
        if len(intensities) > 0:
            intensity_obj = np.mean(np.abs(np.concatenate([intensities, [np.mean(intensities)]]) - self.target_intensity))
        else:
            intensity_obj = abs(self.target_intensity)
        
        # Objective 2: Duration
        if len(durations) > 0:
            duration_obj = np.mean(np.abs(np.concatenate([durations, [np.mean(durations)]]) - self.target_duration)) / 100
        else:
            duration_obj = abs(self.target_duration) / 100
        
        # Objective 3: Frequency
        if frequency < self.target_frequency:
            frequency_obj = 100 * (self.target_frequency - frequency)
        elif frequency > self.target_frequency:
            frequency_obj = np.mean(durations) if len(durations) > 0 else self.target_duration
        else:
            frequency_obj = 0
        
        # Objective 4: Autocorrelation
        synthetic_autocorr = self._calculate_autocorrelation(synthetic, 12)
        historical_autocorr = self.historical_stats[site]['autocorr']
        autocorr_obj = np.mean(np.abs(historical_autocorr - synthetic_autocorr))
        
        # Objective 5: Non-drought distribution
        non_drought_mask = np.ones(len(ssi), dtype=bool)
        for start, end in drought_periods:
            non_drought_mask[start:end+1] = False
        
        if np.any(non_drought_mask):
            non_drought_ssi = ssi[non_drought_mask]
            synth_percentiles = np.percentile(non_drought_ssi, [25, 50, 75])
            hist_percentiles = self.historical_stats[site]['non_drought_percentiles']
            non_drought_obj = np.mean(np.abs(synth_percentiles - hist_percentiles))
        else:
            non_drought_obj = 1.0
        
        # Weighted combination
        total_obj = (self.weights[0] * intensity_obj + 
                    self.weights[1] * duration_obj +
                    self.weights[2] * frequency_obj +
                    self.weights[3] * autocorr_obj +
                    self.weights[4] * non_drought_obj)
        
        return total_obj
    
    def _accept_swap(self, obj_current: float, obj_candidate: float, temperature: float) -> bool:
        """Determine whether to accept a swap based on simulated annealing criterion."""
        if obj_candidate <= obj_current:
            return True
        
        if obj_current > 0:
            prob = np.exp(((obj_current - obj_candidate) / obj_current) / temperature)
            return np.random.rand() < prob
        
        return False
    
    def _format_results(self, results: Dict, n_scenarios: int) -> pd.DataFrame:
        """Format generation results into DataFrame."""
        n_months = self.nyears_generate * 12
        dates = pd.date_range(start='2000-01-01', periods=n_months, freq='M')
        
        formatted_data = {}
        for site in results:
            for scenario in range(n_scenarios):
                col_name = f"{site}_scenario_{scenario + 1}"
                formatted_data[col_name] = results[site][scenario]
        
        return pd.DataFrame(formatted_data, index=dates)