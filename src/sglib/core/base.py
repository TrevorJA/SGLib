"""
Improved Base Generator Class for SGLib

This module provides an abstract base class for all synthetic generation methods.
"""

from abc import ABC, abstractmethod
import logging
import warnings
from typing import Union, Dict, Any, Optional, List, Tuple
import numpy as np
import pandas as pd
from dataclasses import dataclass, field

@dataclass
class GeneratorState:
    is_preprocessed: bool = False
    is_fitted: bool = False


class Generator(ABC):
    """
    Abstract base class for all synthetic generation methods.
    
    All generator implementations should inherit from this class.
    """
    
    @abstractmethod
    def __init__(self,
                 name: Optional[str] = None,
                 debug: bool = False,
                 ) -> None:
        """
        Initialize the generator base class.
        
        Parameters
        ----------
        name : str, optional
            Name identifier for this generator instance
        debug : bool, default False
            Enable debug logging
        """
        self.name = name or self.__class__.__name__
        self.debug = debug
        
        self.state = GeneratorState()
        
        # Setup logging
        self._setup_logging(debug)
        
        
    def _setup_logging(self, 
                       debug: bool) -> None:
        """Setup logging infrastructure."""
        self.logger = logging.getLogger(f"sglib.{self.name}")
        if debug:
            self.logger.setLevel(logging.DEBUG)
        else:
            self.logger.setLevel(logging.INFO)        
            

    def validate_input_data(self, data: Union[pd.Series, pd.DataFrame]) -> pd.DataFrame:
        """
        Validate and standardize input data format.
        
        Parameters
        ----------
        data : pd.Series or pd.DataFrame
            Input time series data
            
        Returns
        -------
        pd.DataFrame
            Validated and standardized data
            
        Raises
        ------
        ValueError
            If data format is invalid
        TypeError
            If data type is unsupported
        """
        # Type checking
        if not isinstance(data, (pd.Series, pd.DataFrame)):
            raise TypeError(
                f"Input data must be pandas Series or DataFrame, got {type(data)}"
            )
        
        # Convert Series to DataFrame
        if isinstance(data, pd.Series):
            data = data.to_frame(name=data.name or 'flow')
        
        # Index validation
        if not isinstance(data.index, pd.DatetimeIndex):
            
            # try to convert index to DatetimeIndex
            try:
                data.index = pd.to_datetime(data.index)
            except Exception as e:
                raise ValueError(
                    f"Data index must be a DatetimeIndex, got {type(data.index)}: {e}"
                )
        
        self.logger.info(
            f"Validated data: {data.shape[0]} timesteps, {data.shape[1]} sites, "
            f"period {data.index[0]} to {data.index[-1]}"
        )
        
        return data


    @abstractmethod
    def preprocessing(self, **kwargs):
        pass

    @abstractmethod
    def fit(self, **kwargs):
        pass

    @abstractmethod
    def generate(self, **kwargs):
        pass
