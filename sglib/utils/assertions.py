


def _check_attr(self, attr):
    """
    Check if attribute of object exists.
    """
    return hasattr(self, attr)


def get_attr(self, attr):
    """
    Get attribute of object.
    """
    is_attr = _check_attr(self, attr)
    if not is_attr:
        raise AttributeError("Object does not have attribute {}.".format(attr))
    else:
        return getattr(self, attr)


def set_attr(self, attr, value):
    """
    Set attribute of object to value.
    """
    setattr(self, attr, value)
    return


def _validate_kwargs(self, default_kwargs, **kwargs):
    """
    Validate kwargs
    """
    if len(kwargs) == 0:
        return
    for key, value in kwargs.items():
        if key in default_kwargs.keys():
            if type(value) != type(default_kwargs[key]):
                raise TypeError("Invalid type for keyword argument {}.".format(key))
        else:
            
            raise KeyError(f"Invalid keyword argument {key}. Options: {default_kwargs.keys()}")
        
    return 




def set_model_kwargs(self, default_kwargs, **kwargs):
    """
    Set model kwargs.
    """
    # Check that all kwargs are valid
    _validate_kwargs(self, default_kwargs, **kwargs)
    
    # Set values for kwargs; if not provided, use default
    for default_key, default_value in default_kwargs.items():
        # Set as user provided value if included
        if default_key in kwargs.keys():
            set_attr(self, default_key, kwargs[default_key])
        else:
            print(f"Using default value {default_value} for kwarg {default_key}.")
            set_attr(self, default_key, default_value)
    return


def update_model_kwargs(self, default_kwargs, **kwargs):
    """
    Update model kwargs based on new values passed to a method. 
    If the attr is not in the default kwargs, raise an error.
    If the attr already exists, update it. 
    Else, set it to the default value.
    """
    
    # Check that all kwargs are valid
    _validate_kwargs(self, default_kwargs, **kwargs)
    
    # Check it it exists
    for key, value in kwargs.items():
        if _check_attr(self, key):
            set_attr(self, key, value)
        elif key in default_kwargs.keys():
            set_attr(self, key, value)
        else:
            raise AttributeError(f"Object does not have attribute {key}.")
