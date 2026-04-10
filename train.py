import atexit
import os
import pathlib
import sys
import warnings

import hydra
import torch

import tools
from buffer import Buffer
from dreamer import Dreamer
from envs import make_envs
from trainer import OnlineTrainer

warnings.filterwarnings("ignore")
sys.path.append(str(pathlib.Path(__file__).parent))
# torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high")


def _maybe_init_wandb(config, logdir):
    """Optionally mirror tensorboard scalars to W&B.

    Activated when the WANDB_PROJECT env var is set. Auth is read from
    ~/.netrc (`wandb login`). Run name defaults to the basename of the
    logdir; override via WANDB_RUN_NAME. The full Hydra config is
    uploaded as W&B config so runs are self-describing.
    """
    project = os.environ.get("WANDB_PROJECT")
    if not project:
        return None
    try:
        import wandb
        from omegaconf import OmegaConf
    except ImportError:
        print("WANDB_PROJECT set but wandb not installed; skipping.")
        return None
    run_name = os.environ.get("WANDB_RUN_NAME") or pathlib.Path(logdir).name
    cfg_dict = OmegaConf.to_container(config, resolve=True)
    return wandb.init(
        project=project,
        name=run_name,
        dir=str(logdir),
        config=cfg_dict,
        sync_tensorboard=True,
        reinit=True,
    )


@hydra.main(version_base=None, config_path="configs", config_name="configs")
def main(config):
    tools.set_seed_everywhere(config.seed)
    if config.deterministic_run:
        tools.enable_deterministic_run()
    logdir = pathlib.Path(config.logdir).expanduser()
    logdir.mkdir(parents=True, exist_ok=True)

    # Mirror stdout/stderr to a file under logdir while keeping console output.
    console_f = tools.setup_console_log(logdir, filename="console.log")
    atexit.register(lambda: console_f.close())

    print("Logdir", logdir)

    wandb_run = _maybe_init_wandb(config, logdir)
    if wandb_run is not None:
        atexit.register(lambda: wandb_run.finish())
        print(f"W&B run: {wandb_run.name}  url={wandb_run.get_url()}")

    logger = tools.Logger(logdir)
    # save config
    logger.log_hydra_config(config)

    replay_buffer = Buffer(config.buffer)

    print("Create envs.")
    train_envs, eval_envs, obs_space, act_space = make_envs(config.env)

    print("Simulate agent.")
    agent = Dreamer(
        config.model,
        obs_space,
        act_space,
    ).to(config.device)

    policy_trainer = OnlineTrainer(config.trainer, replay_buffer, logger, logdir, train_envs, eval_envs)
    policy_trainer.begin(agent)

    items_to_save = {
        "agent_state_dict": agent.state_dict(),
        "optims_state_dict": tools.recursively_collect_optim_state_dict(agent),
    }
    torch.save(items_to_save, logdir / "latest.pt")


if __name__ == "__main__":
    main()
