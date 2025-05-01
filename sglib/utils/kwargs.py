
common_default_kwargs = {
    'deseasonalize': False,
    'standardize': False,
    'normalize': False,
    'log_transform': False,
    'n_realizations': 1,
    'n_timesteps': 100,
    'timestep': 'MS',
    '_is_fit': False,
    '_is_preprocessed': False,
    'verbose': False,
}


default_plot_kwargs = {
    'kind': 'line',
    'figsize': (12, 8),
    'title': None,
    'xlabel': None,
    'ylabel': None,
    'legend': True,
    'legend_loc': 'best',
    'legend_ncol': 1,
    'legend_fontsize': 12,
    'linestyle': '-',
    'linewidth': 2,
    'alpha': 1}


hmm_default_kwargs = common_default_kwargs.copy()
hmm_default_kwargs.update({
    'n_hidden_states': 2,
    'max_iter': 100,
    'tolerance': 0.00000001,
})
