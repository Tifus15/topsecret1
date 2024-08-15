from experiment_launcher import Launcher, is_local

LOCAL = is_local()
TEST = False
USE_CUDA = False

N_SEEDS = 1

if LOCAL:
    N_EXPS_IN_PARALLEL = 1
else:
    N_EXPS_IN_PARALLEL = 1

N_CORES = N_EXPS_IN_PARALLEL
MEMORY_SINGLE_JOB = 8000
MEMORY_PER_CORE = N_EXPS_IN_PARALLEL * MEMORY_SINGLE_JOB // N_CORES
PARTITION = 'amd2,amd'  # 'amd', 'rtx'
GRES = 'gpu:1' if USE_CUDA else None  # gpu:rtx2080:1, gpu:rtx3080:1
CONDA_ENV = 'hnn'  # None

launcher = Launcher(
    exp_name='nbody_master',
    exp_file='main',
    project_name='project02183',  # for hrz cluster
    n_seeds=N_SEEDS,
    n_exps_in_parallel=N_EXPS_IN_PARALLEL,
    n_cores=N_CORES,
    memory_per_core=MEMORY_PER_CORE,
    days=1,
    hours=6,
    minutes=0,
    seconds=0,
    partition=PARTITION,
    conda_env=CONDA_ENV,
    gres=GRES,
    use_timestamp=True,
    compact_dirs=False
)

config_files_l = [
    'configs/nbody.yaml'
]

# Optional arguments for Weights and Biases
wandb_options = dict(
    wandb_enabled=True,  # If True, runs wandb. Default is False.
    wandb_entity='andridenis',
    wandb_project='nbody'
)


launcher.add_experiment(
    # A subdirectory will be created for parameters with a trailing double underscore.
    config__='configs/nbody.yaml',

    config_file_path=config_files_l[0],

    debug=False,

    **wandb_options,
    wandb_group=f'test_group-nbody'
    )

launcher.run(LOCAL, TEST)