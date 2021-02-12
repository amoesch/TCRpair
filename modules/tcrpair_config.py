import os

# Data paths and dirs
project_path = ''
datasets_dir = 'datasets'
datasets_path = os.path.join(project_path, datasets_dir)
log_dir = 'logs'
log_ml_path = os.path.join(project_path, log_dir)
data_dir = 'data'
df_dir = 'dataframes'

models_path = os.path.join(project_path, 'models')

vdjdb_dir = 'vdjdb'
vdjdb_path = os.path.join(datasets_path, vdjdb_dir)
vdjdb_data_path = os.path.join(vdjdb_path, data_dir)
vdjdb_df_path = os.path.join(vdjdb_path, df_dir)

iedb_dir = 'iedb'
iedb_path = os.path.join(datasets_path, iedb_dir)
iedb_data_path = os.path.join(iedb_path, data_dir)
iedb_df_path = os.path.join(iedb_path, df_dir)

imgt_dir = 'imgt'
imgt_path = os.path.join(datasets_path, imgt_dir)
imgt_data_path = os.path.join(imgt_path, data_dir)

td_dir = 'trainingdata'
td_path = os.path.join(datasets_path, td_dir)

val_path = os.path.join(datasets_path, 'validationdata')

plots_dir = 'plots'
plots_path = os.path.join(project_path, plots_dir)
mp_plots_path = os.path.join(plots_dir, 'model_performances')
pr_plots_path = os.path.join(plots_dir, 'predictions')

# VDJdb columns
antigen_cols = ['antigen.epitope', 'antigen.gene', 'antigen.species']
alpha_cols   = ['cdr3.alpha', 'v.alpha', 'j.alpha', 'full.alpha', 'part.alpha']
beta_cols    = ['cdr3.beta', 'v.beta', 'j.beta', 'full.beta', 'part.beta']
seq_cols     = ['cdr3.alpha', 'full.alpha', 'cdr3.beta', 'full.beta', 'antigen.epitope', 'part.alpha', 'part.beta']

# Other
chains = {'A' : 'alpha', 'B' : 'beta'}
mask_value = 999
