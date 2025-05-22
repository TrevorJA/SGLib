import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

class NowakDisaggregator:
    """
    Temporal disaggregation as described in Nowak et al. (2010).
    
    Extended to support multisite disaggregation from monthly to daily streamflows.
    
    For each month in synthetic data, find the N historic monthly flow profiles
    which have similar total flow at the index gauge (sum of all sites). 
    
    Then, randomly select one of the N profiles and use the daily flow proportions
    from that month to disaggregate the synthetic monthly flow at all sites.

    When disaggregating a month, only consider historic profiles from the same 
    month of interest.  
    
    We consider +/- 1-7 days around each month of interest. 
    E.g., When disaggregating flow for April, we consider the 14 monthly flow 
    profiles calculated +/- 7 days around April, and for all years in the historic
    record. 
    
    
    References:
    Nowak, K., Prairie, J., Rajagopalan, B., & Lall, U. (2010). 
    A nonparametric stochastic approach for multisite disaggregation of 
    annual to daily streamflow. Water Resources Research, 46(8).
    """
    
    def __init__(self, 
                 Qh_daily,
                 n_neighbors=5,
                 max_month_shift=7):
        """
        Initialize the NowakDisaggregation class.
        
        This now supports both single site (Series) and multi-site (DataFrame) disaggregation.
        For multi-site, the input should be a DataFrame with sites as columns.
        
        Parameters
        ----------
        Qh_daily : pd.Series or pd.DataFrame
            Daily streamflow data for the historic period. 
            The index should be a datetime index.
            If DataFrame, columns represent different sites.
        n_neighbors : int
            The number of neighbors to use for disaggregation.
        max_month_shift : int
            Maximum number of days to shift around each month center.
        """
        
        self.Qh_daily = self._verify_input_data(Qh_daily)
        self.is_multisite = isinstance(self.Qh_daily, pd.DataFrame)
        
        if self.is_multisite:
            self.n_sites = self.Qh_daily.shape[1]
            self.site_names = list(self.Qh_daily.columns)
            # Create index gauge as sum of all sites
            self.Qh_index = self.Qh_daily.sum(axis=1)
        else:
            self.n_sites = 1
            self.site_names = [self.Qh_daily.name if self.Qh_daily.name else 'site_1']
            self.Qh_index = self.Qh_daily
        
        self.Qs_monthly = None # provided later
        
        self.n_neighbors = n_neighbors
        self.max_month_shift = max_month_shift
        
        # Get historic datetime stats
        self.n_historic_years = self.Qh_index.index.year.nunique()
        self.historic_years = self.Qh_index.index.year.unique()
        
        # dict containing trained KNN models for each month
        self.knn_models = {}
        
        ## Utilities
        # num of days in each month, starting Jan 
        self.days_per_month = [31, 28, 31, 
                               30, 31, 30, 
                               31, 31, 30, 
                               31, 30, 31]
        
    def preprocessing(self):
        pass
    
    def fit(self):
        """
        Fit the NowakDisaggregator to the data.
        
        This will create a dataset of candidate monthly flow profiles for each month,
        and train KNN models for each month.
        """
        
        # Create the dataset of candidate monthly flow profiles
        self.monthly_cumulative_flows, self.daily_flow_profiles = self._make_historic_monthly_profile_dataset()
        
        # Train KNN models for each month
        for month in range(1, 13):
            self._train_knn_model(month)
            
    def generate(self):
        pass
    
    def _verify_input_data(self, Q):
        """
        Checks that the input is:
        - a pandas Series or DataFrame
        - has a datetime index
        
        Parameters
        ----------
        Q : pd.Series or pd.DataFrame
            The input data to check.
        
        Returns
        -------
        Q : pd.Series or pd.DataFrame
            The input data as validated.
        """
        
        if not isinstance(Q, (pd.Series, pd.DataFrame)):
            raise ValueError("Q must be a pandas Series or DataFrame.")
        
        if not isinstance(Q.index, pd.DatetimeIndex):
            raise ValueError(f"Q must have a datetime index. Has {type(Q.index)}.")
        
        return Q
    
    def _make_historic_monthly_profile_dataset(self):
        """
        Create dataset of candidate monthly flow profiles for each month.
        
        For each month, we will have a dataset of monthly flow profiles
        for each year in the historic record, and for +/- max_month_shift days around the month.
    
        This will generate both:
        - dataset of total monthly flows (index gauge), used to find KNN indices
        - dataset of daily flow proportions for each site, used to disaggregate monthly flows
    
        Format:
        monthly_cumulative_flows : dict
            values are np.array of total flows (index gauge) for each year and shift 
            (length = n_historic_years * (2*max_month_shift + 1))
        daily_flow_profiles : dict
            For single site: values are np.array of daily flow proportions for each year and shift 
            (shape = (n_historic_years * (2*max_month_shift + 1), n_days_in_month))
            For multisite: values are np.array of daily flow proportions for each site, year and shift
            (shape = (n_historic_years * (2*max_month_shift + 1), n_days_in_month, n_sites))
        """
        
        # Create a dict to hold monthly cumulative flows and daily profiles
        monthly_cumulative_flows = {}
        daily_flow_profiles = {}
        
        # Make a copy of data with wrap-around datetime to account +/- max_month_shift day shifts
        start_date = self.Qh_index.index[0]
        end_date = self.Qh_index.index[-1]
        wrap_start_date = start_date - pd.DateOffset(days=self.max_month_shift)
        wrap_end_date = end_date + pd.DateOffset(days=self.max_month_shift)
            
        # Create wrapped index gauge
        Qh_index_wrap = pd.Series(index=pd.date_range(start=wrap_start_date,
                                                     end=wrap_end_date, 
                                                     freq='D'))
        Qh_index_wrap = Qh_index_wrap.astype(float)
        
        Qh_index_wrap.loc[wrap_start_date:start_date] = self.Qh_index.loc[end_date - pd.DateOffset(days=self.max_month_shift):end_date]
        Qh_index_wrap.loc[start_date:end_date] = self.Qh_index.loc[start_date:end_date]
        Qh_index_wrap.loc[end_date:wrap_end_date] = self.Qh_index.loc[start_date:start_date + pd.DateOffset(days=self.max_month_shift)]
        
        # forward and backward fill the NaN values
        Qh_index_wrap = Qh_index_wrap.ffill().bfill()
        
        # Create wrapped data for all sites
        if self.is_multisite:
            Qh_daily_wrap = pd.DataFrame(index=pd.date_range(start=wrap_start_date,
                                                           end=wrap_end_date, 
                                                           freq='D'),
                                       columns=self.site_names)
            Qh_daily_wrap = Qh_daily_wrap.astype(float)
            
            Qh_daily_wrap.loc[wrap_start_date:start_date] = self.Qh_daily.loc[end_date - pd.DateOffset(days=self.max_month_shift):end_date]
            Qh_daily_wrap.loc[start_date:end_date] = self.Qh_daily.loc[start_date:end_date]
            Qh_daily_wrap.loc[end_date:wrap_end_date] = self.Qh_daily.loc[start_date:start_date + pd.DateOffset(days=self.max_month_shift)]
            
            # forward and backward fill the NaN values
            Qh_daily_wrap = Qh_daily_wrap.ffill().bfill()
        else:
            Qh_daily_wrap = Qh_index_wrap.copy()
        
        # Loop through each month
        for month in range(1, 13):
            
            # Array of cumulative flow (index gauge)
            monthly_cumulative_flows[month] = np.zeros(shape=(self.n_historic_years * (2 * self.max_month_shift + 1),))
            
            # Array of daily flow proportions
            if self.is_multisite:
                daily_flow_profiles[month] = np.zeros(shape=(self.n_historic_years * (2 * self.max_month_shift + 1), 
                                                           self.days_per_month[month - 1],
                                                           self.n_sites))
            else:
                daily_flow_profiles[month] = np.zeros(shape=(self.n_historic_years * (2 * self.max_month_shift + 1), 
                                                           self.days_per_month[month - 1]))
            
            # loop through time shifts
            for shift in range(-self.max_month_shift, self.max_month_shift + 1):
                
                # Loop through each year
                for y, year in enumerate(self.historic_years):
                    
                    # Get the start and end dates for the 'month' (accounting for shift)
                    start_date = pd.Timestamp(year=year, month=month, day=1) + pd.DateOffset(days=shift)
                    end_date = start_date + pd.DateOffset(days=self.days_per_month[month - 1]-1)
                    
                    # Get the daily flow data for the month (index gauge)
                    daily_index_data = Qh_index_wrap.loc[start_date:end_date]
                    
                    # Calculate the total monthly flow (index gauge)
                    total_monthly_flow = daily_index_data.sum()
                    
                    # index for this month value
                    idx = y * (2 * self.max_month_shift + 1) + (shift + self.max_month_shift)

                    # Store the total monthly flow (index gauge)
                    monthly_cumulative_flows[month][idx] = total_monthly_flow
                    
                    # Store the daily flow proportions for each site
                    if self.is_multisite:
                        daily_site_data = Qh_daily_wrap.loc[start_date:end_date]
                        for s, site in enumerate(self.site_names):
                            site_total = daily_site_data[site].sum()
                            if site_total > 0:
                                daily_flow_profiles[month][idx, :, s] = daily_site_data[site].values / site_total
                            else:
                                # Handle zero flow case
                                daily_flow_profiles[month][idx, :, s] = 1.0 / len(daily_site_data)
                            
                            # Ensure proportions are valid
                            daily_flow_profiles[month][idx, :, s] = np.clip(daily_flow_profiles[month][idx, :, s], 0, 1)
                            # Renormalize to ensure they sum to 1
                            prop_sum = daily_flow_profiles[month][idx, :, s].sum()
                            if prop_sum > 0:
                                daily_flow_profiles[month][idx, :, s] /= prop_sum
                    else:
                        if total_monthly_flow > 0:
                            daily_flow_profiles[month][idx, :] = daily_index_data.values / total_monthly_flow
                        else:
                            # Handle zero flow case
                            daily_flow_profiles[month][idx, :] = 1.0 / len(daily_index_data)
                        
                        # limit daily flow proportions to [0, 1]
                        daily_flow_profiles[month][idx, :] = np.clip(daily_flow_profiles[month][idx, :], 0, 1)
                        # Renormalize to ensure they sum to 1
                        prop_sum = daily_flow_profiles[month][idx, :].sum()
                        if prop_sum > 0:
                            daily_flow_profiles[month][idx, :] /= prop_sum
    
        self.monthly_cumulative_flows = monthly_cumulative_flows
        self.daily_flow_profiles = daily_flow_profiles
        return monthly_cumulative_flows, daily_flow_profiles
    
    def _train_knn_model(self, 
                        month,
                        n_neighbors=None):
        """
        Train a KNN model for the given month.
        
        KNN is based on the index gauge (sum of all sites) flows.
        
        Parameters
        ----------
        month : int
            The month to train the model for (1-12).
        n_neighbors : int
            The number of neighbors to use for the model.
        
        Returns
        -------
        knn : NearestNeighbors
            The trained KNN model.
        """
        
        if n_neighbors is None:
            n_neighbors = self.n_neighbors
        
        # Check if the model is already trained
        if month in self.knn_models:
            return self.knn_models[month]
        else:
            # Get the historic flows (index gauge) which are in the same month
            historic_flows = self.monthly_cumulative_flows[month]
            
            # historic_flows is a 1D array of total flows for each year and shift
            # reshape to 2D array for KNN
            historic_flows = historic_flows.reshape(-1, 1)
            
            # Create the KNN model
            knn = NearestNeighbors(n_neighbors=n_neighbors)
            
            # Fit the model to the historic flows
            knn.fit(historic_flows)
            
            # Store the model in the dict
            self.knn_models[month] = knn
            
            return knn
    
    def find_knn_indices(self, 
                        Qs_monthly_array, 
                        month,
                        n_neighbors=None):
        """
        Given cumulative monthly flow values, find the K nearest neighbors
        from the historic dataset.
        
        Parameters
        ----------
        Qs_monthly_array : np.array
            The cumulative monthly flow values for the month to disaggregate.
        month : int
            The calendar month which is being disaggregated (1-12).
        n_neighbors : int
            The number of neighbors to find.
        
        Returns
        -------
        distances : np.array
            The distances to the K nearest neighbors.
        indices : np.array
            The indices of the K nearest neighbors in the historic dataset.
        """
        
        if n_neighbors is None:
            n_neighbors = self.n_neighbors
                    
        # Qs_monthly_array is a 1D array of total flows for each month in the synthetic dataset
        # reshape to 2D array for KNN
        Qs_monthly_array = Qs_monthly_array.reshape(-1, 1)
        
        # get the KNN model for the month
        knn = self._train_knn_model(month, n_neighbors)
        
        # get the indices of the K nearest neighbors
        distances, indices = knn.kneighbors(Qs_monthly_array)
        
        return distances, indices
    
    def sample_knn_monthly_flows(self,
                                Qs_monthly_array, 
                                month,
                                n_neighbors=None,
                                sample_method='distance_weighted'): 
        """
        Given cumulative monthly flow values, sample K nearest neighbors
        from the historic dataset.
        
        Parameters
        ----------
        Qs_monthly_array : np.array
            The cumulative monthly flow values for the month to disaggregate.
        month : int
            The calendar month which is being disaggregated (1-12).
        n_neighbors : int
            The number of neighbors to sample.
        sample_method : str
            The sampling method to use.
        
        Returns
        -------
        sampled_indices : np.array
            The sampled indices from the historic dataset.
        """
        
        if n_neighbors is None:
            n_neighbors = self.n_neighbors
        
        # get the K nearest neighbors
        distances, indices = self.find_knn_indices(Qs_monthly_array, month, n_neighbors)
        
        # sample a single index
        if sample_method == 'distance_weighted':
            sampled_indices = []
            for i in range(indices.shape[0]):            
                # sample based on distance
                weights = 1 / (distances[i,:] + 1e-10)  # Add small epsilon to avoid division by zero
                weights = weights / weights.sum()
                sampled_indices.append(np.random.choice(indices[i,:].flatten(), p=weights))
                
        elif sample_method == 'lall_and_sharma_1996':
            weights = []
            sampled_indices = []
            denom = np.array([1/i for i in range(1, n_neighbors+1)]).sum()
            for i in range(1, n_neighbors+1):
                w = 1/i / denom
                weights.append(w)
            weights = np.array(weights)
            
            for i in range(indices.shape[0]):
                sampled_indices.append(np.random.choice(indices[i,:].flatten(), p=weights))
        else:
            raise ValueError("Invalid sample method. Must be 'distance_weighted' or 'lall_and_sharma_1996'.")
        
        return np.array(sampled_indices)
        
    def disaggregate_monthly_flows(self, 
                                  Qs_monthly,
                                  n_neighbors=None,
                                  sample_method='distance_weighted'):
        """
        Disaggregate monthly to daily flows using the Nowak method.
        
        Parameters
        ----------
        Qs_monthly : pd.Series or pd.DataFrame
            Monthly streamflow data for the synthetic period. 
            The index should be a datetime index.
            For multisite, should be DataFrame with same columns as historic data.
        n_neighbors : int
            The number of neighbors to use for disaggregation. 
        sample_method : str
            The method to use for sampling the K nearest neighbors. 
        
        Returns
        -------
        Qs_daily : pd.Series or pd.DataFrame
            Daily streamflow data for the synthetic period. 
            The index will be a datetime index.
            For multisite, returns DataFrame with same columns as input.
        """
        
        if n_neighbors is None:
            n_neighbors = self.n_neighbors
        
        Qs_monthly = self._verify_input_data(Qs_monthly)
        
        # Check if multisite consistency
        if self.is_multisite:
            if not isinstance(Qs_monthly, pd.DataFrame):
                raise ValueError("For multisite disaggregation, Qs_monthly must be a DataFrame.")
            if not all(col in self.site_names for col in Qs_monthly.columns):
                raise ValueError("Qs_monthly columns must match the historic data columns.")
            # Create index gauge for synthetic data
            Qs_monthly_index = Qs_monthly.sum(axis=1)
        else:
            if isinstance(Qs_monthly, pd.DataFrame):
                if Qs_monthly.shape[1] != 1:
                    raise ValueError("For single site disaggregation, Qs_monthly must be a Series or single-column DataFrame.")
                Qs_monthly = Qs_monthly.iloc[:, 0]
            Qs_monthly_index = Qs_monthly
        
        syn_start_date = Qs_monthly.index[0]
        syn_end_date = Qs_monthly.index[-1]
        syn_years = Qs_monthly.index.year.unique()
        
        # Setup output
        daily_index = pd.date_range(start=Qs_monthly.index[0], 
                                   end=Qs_monthly.index[-1] + pd.offsets.MonthEnd(0), 
                                   freq='D')
        
        if self.is_multisite:
            Qs_daily = pd.DataFrame(index=daily_index, columns=self.site_names)
            Qs_daily = Qs_daily.astype(float)
        else:
            Qs_daily = pd.Series(index=daily_index)
            Qs_daily = Qs_daily.astype(float)
        
        Qs_daily[:] = np.nan

        # loop through months
        for month in range(1, 13):
            
            monthly_mask = Qs_monthly_index.index.month == month
            
            if not monthly_mask.any():
                continue
            
            # Get the monthly flow for the month (index gauge)
            Qs_monthly_index_array = Qs_monthly_index[monthly_mask].values
            
            # Get the K nearest neighbors
            sampled_indices = self.sample_knn_monthly_flows(Qs_monthly_index_array, month, n_neighbors, sample_method)
            
            # Get the daily flow proportions for the sampled indices
            if self.is_multisite:
                daily_flow_proportions = self.daily_flow_profiles[month][sampled_indices, :, :]  # shape: (n_years, n_days, n_sites)
            else:
                daily_flow_proportions = self.daily_flow_profiles[month][sampled_indices, :]  # shape: (n_years, n_days)
            
            # For each year, disaggregate the Qs_monthly using the sampled daily flow proportions
            month_dates = Qs_monthly_index.index[monthly_mask]
            
            for y, month_date in enumerate(month_dates):
                # Get the start and end dates for the month
                start_date = month_date
                end_date = start_date + pd.offsets.MonthEnd(0)
                
                # Get the daily flow proportions for the month
                if self.is_multisite:
                    daily_flow_proportions_for_month = daily_flow_proportions[y, :, :]  # shape: (n_days, n_sites)
                else:
                    daily_flow_proportions_for_month = daily_flow_proportions[y, :]  # shape: (n_days,)
                
                # Handle leap year adjustments
                expected_days = len(Qs_daily.loc[start_date:end_date])
                if self.is_multisite:
                    actual_days = daily_flow_proportions_for_month.shape[0]
                else:
                    actual_days = len(daily_flow_proportions_for_month)
                
                if actual_days != expected_days:
                    if expected_days == actual_days + 1:  # Leap year case
                        if self.is_multisite:
                            # Add one more day (repeat last day)
                            last_day = daily_flow_proportions_for_month[-1:, :]
                            daily_flow_proportions_for_month = np.vstack([daily_flow_proportions_for_month, last_day])
                            # Renormalize each site
                            for s in range(self.n_sites):
                                site_sum = daily_flow_proportions_for_month[:, s].sum()
                                if site_sum > 0:
                                    daily_flow_proportions_for_month[:, s] /= site_sum
                        else:
                            # Add one more day (repeat last day)
                            daily_flow_proportions_for_month = np.append(daily_flow_proportions_for_month, 
                                                                       daily_flow_proportions_for_month[-1])
                            # Renormalize
                            daily_flow_proportions_for_month = daily_flow_proportions_for_month / daily_flow_proportions_for_month.sum()
                    else:
                        print(f"Warning: Daily flow proportions for month {month} and year {month_date.year} "
                              f"with length {actual_days} do not match the expected length of {expected_days}.")
                
                # Disaggregate the monthly flow using the daily flow proportions
                if self.is_multisite:
                    for s, site in enumerate(self.site_names):
                        site_monthly_flow = Qs_monthly.loc[month_date, site]
                        Qs_daily.loc[start_date:end_date, site] = site_monthly_flow * daily_flow_proportions_for_month[:, s]
                else:
                    monthly_flow = Qs_monthly_index_array[y]
                    Qs_daily.loc[start_date:end_date] = monthly_flow * daily_flow_proportions_for_month
                
                # Check for issues
                if self.is_multisite:
                    for site in self.site_names:
                        site_data = Qs_daily.loc[start_date:end_date, site]
                        if site_data.isnull().any() or (site_data < 0).any():
                            msg = f"Disaggregation failed for site {site}, month {month} and year {month_date.year}. "
                            msg += f"Qs_daily contains NaN or negative values."
                            raise ValueError(msg)
                else:
                    daily_data = Qs_daily.loc[start_date:end_date]
                    if daily_data.isnull().any() or (daily_data < 0).any():
                        msg = f"Disaggregation failed for month {month} and year {month_date.year}. "
                        msg += f"Qs_daily contains NaN or negative values."
                        raise ValueError(msg)
        
        return Qs_daily

# import numpy as np
# import pandas as pd
# from sklearn.neighbors import NearestNeighbors

# class NowakDisaggregator:
#     """
#     Temporal disaggregation as described in Nowak et al. (2010).
    
#     Used to disaggregate monthly streamflows to daily streamflows.
    
#     For each month in synthetic data, find the N historic monthly flow profiles
#     which have similar total flow. 
    
#     Then, randomly select one of the N profiles and use the daily flow proportions
#     from that month to disaggregate the synthetic monthly flow.

#     When disaggregating a month, only consider historic profiles from the same 
#     month of interest.  
    
#     We consider +/- 1-7 days around each month of interest. 
#     E.g., When disaggregating flow for April, we consider the 14 monthly flow 
#     profiles calculated +/- 7 days around April, and for all years in the historic
#     record. 
    
    
#     References:
#     Nowak, K., Prairie, J., Rajagopalan, B., & Lall, U. (2010). 
#     A nonparametric stochastic approach for multisite disaggregation of 
#     annual to daily streamflow. Water Resources Research, 46(8).
#     """
    
#     def __init__(self, 
#                  Qh_daily,
#                  n_neighbors=5,
#                  max_month_shift=7):
#         """
#         Initialize the NowakDisaggregation class.
        
#         This is currently setup to work with a single site.
#         The input data should be a single time series of daily streamflow data.
#         The data should be in a pandas DataFrame or Series, with a datetime index.
        
#         Parameters
#         ----------
#         Qh_daily : pd.Series or pd.DataFrame
#             Daily streamflow data for the historic period. 
#             The index should be a datetime index. 
#         n_neighbors : int
#             The number of neighbors to use for disaggregation.
#         """
        
#         self.Qh_daily = self._verify_input_is_series_like(Qh_daily)
#         self.Qs_monthly = None # provided later
        
#         self.n_neighbors = n_neighbors
#         self.max_month_shift = max_month_shift
        
        
#         # Get historic datetime stats
#         self.n_historic_years = self.Qh_daily.index.year.nunique()
#         self.historic_years = self.Qh_daily.index.year.unique()
        
#         # dict containing trained KNN models for each month
#         self.knn_models = {}
        
#         ## Utilities
#         # num of days in each month, starting Jan 
#         self.days_per_month = [31, 28, 31, 
#                                30, 31, 30, 
#                                31, 31, 30, 
#                                31, 30, 31]
        
#     def preprocessing(self):
#         pass
    
    
#     def fit(self):
#         """
#         Fit the NowakDisaggregator to the data.
        
#         This will create a dataset of candidate monthly flow profiles for each month,
#         and train KNN models for each month.
#         """
        
#         # Create the dataset of candidate monthly flow profiles
#         self.monthly_cumulative_flows, self.daily_flow_profiles = self._make_historic_monthly_profile_dataset()
        
#         # Train KNN models for each month
#         for month in range(1, 13):
#             self._train_knn_model(month)
            
    
#     def generate(self):
#         pass
    
    
#     def _verify_input_is_series_like(self, Q):
#         """
#         Checks that the input is:
#         - a pandas Series or DataFrame with a single column
#         - has a datetime index
        
#         Parameters
#         ----------
#         Q : pd.Series or pd.DataFrame
#             The input data to check.
        
#         Returns
#         -------
#         Q : pd.Series
#             The input data as a pandas Series.
#         """
#         # Check if Q is a pandas Series or DataFrame
#         if isinstance(Q, pd.DataFrame):
#             assert Q.shape[1] == 1, "Q must be a pandas Series or DataFrame with a single column."
#             Q = Q.iloc[:, 0]
#         elif not isinstance(Q, pd.Series):
#             raise ValueError("Q must be a pandas Series or DataFrame with a single column.")
        
#         if not isinstance(Q.index, pd.DatetimeIndex):
#             raise ValueError(f"Q must have a datetime index. Has {type(Q.index)}.")
        
#         return Q
    
    
#     def _make_historic_monthly_profile_dataset(self):
#         """
#         Create dataset of candidate monthly flow profiles for each month.
        
#         For each month, we will have a dataset of monthly flow profiles
#         for each year in the historic record, and for +/- 7 days around the month.
    
#         This will generate both:
#         - dataset of total monthly flows, used to find KNN indices
#         - dataset of daily flow proportions, used to disaggregate monthly flows
    
#         Format:
#         monthly_cumulative_flows : dict
#             values are np.array of total flows for each year and shift (length = n_historic_years * 15)
#         daily_flow_profiles : dict
#             values are np.array of daily flow proportions for each year and shift (shape = (n_historic_years * 15, n_days_in_month))
    
#         """
#         # Create a dict to hold monthly cumulative flows and daily profiles
        
#         monthly_cumulative_flows = {}
#         daily_flow_profiles = {}
        
#         # Make a copy of Qh_daily with wrap-around datetime to account +/- 7 day shifts
#         # add last 7 days to start
#         start_date = self.Qh_daily.index[0]
#         end_date = self.Qh_daily.index[-1]
#         wrap_start_date = start_date - pd.DateOffset(days=self.max_month_shift)
#         wrap_end_date = end_date + pd.DateOffset(days=self.max_month_shift)
            
#         Qh_daily_wrap = pd.Series(index=pd.date_range(start=wrap_start_date,
#                                                             end=wrap_end_date, 
#                                                             freq='D'))
#         Qh_daily_wrap = Qh_daily_wrap.astype(float)
        
#         Qh_daily_wrap.loc[wrap_start_date:start_date] = self.Qh_daily.loc[end_date - pd.DateOffset(days=self.max_month_shift):end_date]
#         Qh_daily_wrap.loc[start_date:end_date] = self.Qh_daily.loc[start_date:end_date]
#         Qh_daily_wrap.loc[end_date:wrap_end_date] = self.Qh_daily.loc[start_date:start_date + pd.DateOffset(days=self.max_month_shift)]
        
#         # forward and backward fill the NaN values
#         Qh_daily_wrap = Qh_daily_wrap.ffill().bfill()
        
        
#         # Loop through each month
#         for month in range(1, 13):
            
#             # Array of cumulative flow
#             monthly_cumulative_flows[month] = np.zeros(shape=(self.n_historic_years * (2 * self.max_month_shift + 1),))
            
#             daily_flow_profiles[month] = np.zeros(shape=(self.n_historic_years * (2 * self.max_month_shift + 1), 
#                                                             self.days_per_month[month - 1]))
            
#             # loop through time shifts
#             for shift in range(-self.max_month_shift, self.max_month_shift + 1):
                
#                 # Loop through each year
#                 for y, year in enumerate(self.historic_years):
                    
#                     # Get the start and end dates for the 'month' (accounting for shift)
#                     start_date = pd.Timestamp(year=year, month=month, day=1) + pd.DateOffset(days=shift)
#                     end_date = start_date + pd.DateOffset(days=self.days_per_month[month - 1]-1)
                    
#                     # Get the daily flow data for the month
#                     daily_flow_data = Qh_daily_wrap.loc[start_date:end_date]
                    
#                     # Calculate the total monthly flow
#                     total_monthly_flow = daily_flow_data.sum()
                    
#                     # index for this month value
#                     idx = (y * (shift + self.max_month_shift) + shift)

#                     # Store the total monthly flow
#                     monthly_cumulative_flows[month][idx] = total_monthly_flow
                    
#                     # Store the daily flow proportions
#                     daily_flow_profiles[month][idx, :] = daily_flow_data / total_monthly_flow
                                    
#                     # limit daily flow proportions to 1
#                     daily_flow_profiles[month][idx, :] = np.clip(daily_flow_profiles[month][idx, :], 0, 1)
                
    
#         self.monthly_cumulative_flows = monthly_cumulative_flows
#         self.daily_flow_profiles = daily_flow_profiles
#         return monthly_cumulative_flows, daily_flow_profiles
    
#     def _train_knn_model(self, 
#                             month,
#                             n_neighbors=5):
#         """
#         Train a KNN model for the given month.
        
#         Keeps a dict of KNN models for each month.
#         If already trained for that month, return the model.
        
#         Parameters
#         ----------
#         month : int
#             The month to train the model for (1-12).
#         n_neighbors : int
#             The number of neighbors to use for the model.
        
#         Returns
#         -------
#         knn : NearestNeighbors
#             The trained KNN model.
#         """
        
#         # Check if the model is already trained
#         if month in self.knn_models:
#             return self.knn_models[month]
#         else:
#             # Get the historic flows which are in the same month
#             historic_flows = self.monthly_cumulative_flows[month]
            
#             # historic_flows is a 1D array of total flows for each year and shift
#             # reshape to 2D array for KNN
#             historic_flows = historic_flows.reshape(-1, 1)
            
#             # Create the KNN model
#             knn = NearestNeighbors(n_neighbors=n_neighbors)
            
#             # Fit the model to the historic flows
#             knn.fit(historic_flows)
            
#             # Store the model in the dict
#             self.knn_models[month] = knn
            
#             return knn
    
    
#     def find_knn_indices(self, 
#                             Qs_monthly_array, 
#                             month,
#                             n_neighbors=5):
#         """
#         Given a single cumulative monthly flow value, find the K nearest neighbors
#         from the historic dataset.
        
#         Parameters
#         ----------
#         monthly_cumulative_flow : float
#             The cumulative monthly flow value for the month to disaggregate.
#         month : int
#             The calendar month which is being disaggregated (1-12).
#         n_neighbors : int
#             The number of neighbors to find.
        
#         Returns
#         -------
#         indices : np.array
#             The indices of the K nearest neighbors in the historic dataset.
#         """
                    
#         # Qs_monthly_array is a 1D array of total flows for each month in the synthetic dataset
#         # reshape to 2D array for KNN
#         Qs_monthly_array = Qs_monthly_array.reshape(-1, 1)
        
#         # get the KNN model for the month
#         knn = self._train_knn_model(month, n_neighbors)
        
#         # get the indices of the K nearest neighbors
#         distances, indices = knn.kneighbors(Qs_monthly_array)
        
#         # indices is a 2D array of shape (n_neighbors, n_months)
#         return distances, indices
    
#     def sample_knn_monthly_flows(self,
#                                     Qs_monthly_array, 
#                                     month,
#                                     n_neighbors=5,
#                                     sample_method='distance_weighted'): 
#         """
#         Given a single cumulative monthly flow value, sample K nearest neighbors
#         from the historic dataset.
        
#         Parameters
#         ----------
#         Qs_monthly_array : np.array
#             The cumulative monthly flow values for the month to disaggregate.
#         month : int
#             The calendar month which is being disaggregated (1-12).
#         n_neighbors : int
#             The number of neighbors to sample.
#         """
        
#         # get the K nearest neighbors
#         distances, indices = self.find_knn_indices(Qs_monthly_array, month, n_neighbors)
#         # sample a single index
#         if sample_method == 'distance_weighted':
#             sampled_indices = []
#             for i in range(indices.shape[0]):            
#                 # sample based on distance
#                 weights = 1 / distances[i,:]
#                 weights = weights / weights.sum()
#                 sampled_indices.append(np.random.choice(indices[i,:].flatten(), p=weights))
                
#         elif sample_method == 'lall_and_sharma_1996':
#             weights = []
#             sampled_indices = []
#             denom = np.array([1/i for i in range(1, n_neighbors+1)]).sum()
#             for i in range(1, n_neighbors+1):
#                 w = 1/i / denom
#                 weights.append(w)
#             weights = np.array(weights)
            
#             for i in range(indices.shape[0]):
#                 sampled_indices.append(np.random.choice(indices[i,:].flatten(), p=weights))
            
            
#         else:
#             raise ValueError("Invalid sample method. Must be 'distance_weighted'.")
        
#         return np.array(sampled_indices)
        
#     def disaggregate_monthly_flows(self, 
#                                     Qs_monthly,
#                                     n_neighbors=5,
#                                     sample_method='distance_weighted'):
#         """
#         Disaggregate daily to monthly flows using the Nowak method.
        
#         Parameters
#         ----------
#         Qs_monthly : pd.Series
#             Monthly streamflow data for the synthetic period. 
#             The index should be a datetime index.
#         n_neighbors : int
#             The number of neighbors to use for disaggregation. 
#         sample_method : str
#             The method to use for sampling the K nearest neighbors. 
#             Options are 'distance_weighted'.
        
#         Returns
#         -------
#         Qs_daily : pd.Series 
#             Daily streamflow data for the synthetic period. 
#             The index will be a datetime index.
#         """
        
#         Qs_monthly = self._verify_input_is_series_like(Qs_monthly)
        
#         syn_start_date = Qs_monthly.index[0]
#         syn_end_date = Qs_monthly.index[-1]
#         syn_years = Qs_monthly.index.year.unique()
        
#         # Check that the input data is monthly
#         # if Qs_monthly.index.freq != 'MS':
#         #     raise ValueError(f"Qs_monthly must have a monthly frequency. Has {Qs_monthly.index.freq}.")

#         # Setup output series
#         Qs_daily = pd.Series(index=pd.date_range(start=Qs_monthly.index[0], 
#                                                     end=Qs_monthly.index[-1] + pd.offsets.MonthEnd(0), 
#                                                     freq='D'))
#         Qs_daily = Qs_daily.astype(float)
#         Qs_daily[:] = np.nan
        

#         # loop through months
#         for month in range(1, 13):
            
#             Qs_monthly_indices = Qs_monthly.index.month == month
            
#             # Get the monthly flow for the month
#             Qs_monthly_array = Qs_monthly[Qs_monthly.index.month == month].values
            
#             # Get the K nearest neighbors
#             sampled_indices = self.sample_knn_monthly_flows(Qs_monthly_array, month, n_neighbors, sample_method)
            
#             # Get the daily flow proportions for the sampled indices
#             daily_flow_proportions = self.daily_flow_profiles[month][sampled_indices, :]
            
#             # For each year, disaggregate the Qs_monthly using the sampled
#             # daily flow proportions
#             for y, year in enumerate(syn_years):
#                 # Get the start and end dates for the month
#                 start_date = Qs_monthly.index[Qs_monthly_indices][y]
#                 end_date = start_date + pd.offsets.MonthEnd(0)
                
#                 # Get the daily flow proportions for the month
#                 daily_flow_proportions_for_month = daily_flow_proportions[y, :]
                
#                 # Disaggregate the monthly flow using the daily flow proportions
#                 if len(daily_flow_proportions_for_month) != len(Qs_daily.loc[start_date:end_date]):
#                     print(f"Warning: Daily flow proportions for month {month} and year {year} with length {len(daily_flow_proportions_for_month)} do not match the expected length of {len(Qs_daily.loc[start_date:end_date])}.")
                    
#                     # this is happening because of leap year
#                     # add 1 day to the daily flow proportions
#                     daily_flow_proportions_for_month = np.append(daily_flow_proportions_for_month, daily_flow_proportions_for_month[-1])
                    
#                     # rescale to 1
#                     daily_flow_proportions_for_month = daily_flow_proportions_for_month / daily_flow_proportions_for_month.sum()
                    
                    
#                 Qs_daily.loc[start_date:end_date] = Qs_monthly_array[y] * daily_flow_proportions_for_month
        
        
#                 # check that Qs_daily is not NaN and is >= 0
#                 if Qs_daily.loc[start_date:end_date].isnull().any() or (Qs_daily.loc[start_date:end_date] < 0).any():
#                     msg = f"Disaggregation failed for month {month} and year {year}. Qs_daily contains NaN or negative values."
#                     msg += f"### Qs_daily: {Qs_daily.loc[start_date:end_date]}"
#                     msg += f"### Qs_monthly: {Qs_monthly_array[y]}"
#                     msg += f"### daily_flow_proportions: {daily_flow_proportions_for_month}"
#                     raise ValueError(msg)
                
#         return Qs_daily
                    
                        