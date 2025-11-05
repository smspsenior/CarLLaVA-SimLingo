import os

import hydra
import pytorch_lightning as pl
import torch
from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint
from omegaconf import OmegaConf
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelSummary, ThroughputMonitor
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger, WandbLogger

from simlingo_base_training.callbacks.visualise import VisualiseCallback
from simlingo_base_training.config import TrainConfig
from simlingo_base_training.utils.logging_project import setup_logging

class SimlingoCompatibleCheckpoint(pl.Callback):
    def __init__(self, dirpath="./checkpoints", every_n_epochs=1, save_last=True):
        super().__init__()
        self.dirpath = dirpath
        self.every = every_n_epochs
        self.save_last = save_last
        os.makedirs(self.dirpath, exist_ok=True)

    def _dump(self, trainer: pl.Trainer, pl_module: pl.LightningModule, filename: str):
        ckpt = {
            "state_dict": pl_module.state_dict(),
            "epoch": int(trainer.current_epoch),
            "global_step": int(trainer.global_step),
            "pytorch-lightning_version": pl.__version__,
        }
        path = os.path.join(self.dirpath, filename)
        torch.save(ckpt, path)
        if trainer.is_global_zero:
            print(f"[SimlingoCompatibleCheckpoint] saved -> {path}")

    def on_train_epoch_end(self, trainer, pl_module):
        if not trainer.is_global_zero:
            return
        if (trainer.current_epoch + 1) % self.every == 0:
            self._dump(trainer, pl_module, f"epoch={trainer.current_epoch:03d}.ckpt")
        if self.save_last:
            self._dump(trainer, pl_module, "last.ckpt")

@hydra.main(config_path=f"config", config_name="config", version_base="1.1")
def main(cfg: TrainConfig):
    torch.set_float32_matmul_precision("high")
    pl.seed_everything(cfg.seed, workers=True)

    # turn off wandb uploading
    if cfg.debug:
        os.environ["WANDB_MODE"] = "offline"

    cfg.wandb_name = f"{cfg.wandb_name}_{cfg.name}"

    cfg.model.vision_model.use_global_img = cfg.data_module.use_global_img

    data_module = hydra.utils.instantiate(
        cfg.data_module, 
        encoder_variant=cfg.model.vision_model.variant,
        llm_variant=cfg.model.language_model.variant,
        predict=False
    )
    model = hydra.utils.instantiate(
        cfg.model,
        route_as=cfg.data_module.route_as, 
        vision_model={
            "use_global_img": cfg.data_module.use_global_img,
            }
        )

    if cfg.checkpoint is not None:
        if os.path.isdir(cfg.checkpoint):
            state_dict = get_fp32_state_dict_from_zero_checkpoint(cfg.checkpoint)
        else:
            state_dict = torch.load(cfg.checkpoint, map_location="cpu")
        model.load_state_dict(state_dict)

        
    # print config
    print(OmegaConf.to_yaml(cfg))
    os.environ["WANDB_DISABLE_CODE"] = "True"
    
    if cfg.overfit > 0:
        overfit = cfg.overfit
        
    # setup logging
    setup_logging(cfg)

    # resume training
    resume_path = cfg.resume_path
    resume_wandb = False

    # if folder for this experiment does not exist set resume to true
    # to create necessary folders to resume wandb logging later
    if not os.path.exists(resume_path):
        resume_wandb = True
    elif os.path.exists(resume_path) and cfg.resume:
        resume_wandb = True

    if os.path.exists(resume_path) and cfg.resume:
        resume_path = resume_path
    else:
        resume_path = None

    # setup lightning logger
    loggers = []
    # csvlogger = CSVLogger("log/", "CSVLogger")
    # loggers.append(csvlogger)
    # csvlogger = None

    wandblogger = None
    if not cfg.debug and cfg.enable_wandb:
        wandblogger = WandbLogger(
            project=cfg.wandb_project,
            id=cfg.wandb_name,
            name=cfg.wandb_name,
            config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True),
            resume=resume_wandb,
        )
        wandblogger.watch(model)
        loggers.append(wandblogger)

    strategy = cfg.strategy
    if strategy == "deepspeed_stage_2":
        strategy = pl.strategies.DeepSpeedStrategy(
            stage=2, 
            # stage=3, zero3_init_flag=True,
            loss_scale=cfg.fp16_loss_scale, logging_batch_size_per_gpu=cfg.data_module.batch_size
        )

    sim_ckpt = SimlingoCompatibleCheckpoint(
        dirpath="./checkpoints",
        every_n_epochs=cfg.val_every_n_epochs,
        save_last=True,
    )

    lr_monitor = LearningRateMonitor(logging_interval='step')
    model_summary = ModelSummary(max_depth=3)
    callbacks=[
        sim_ckpt,
        model_summary,
        VisualiseCallback(interval=1000)
    ]
    if not cfg.debug: 
        callbacks.append(lr_monitor)
    
    print(f"Number of GPUS: {cfg.gpus}")
    overfit = 0
    
    if cfg.gpus >= 1:
        trainer = Trainer(
            accelerator="gpu",
            benchmark=True,
            callbacks=callbacks,
            devices=cfg.gpus,
            enable_checkpointing=False,
            gradient_clip_val=1.0,
            log_every_n_steps=20,
            logger=loggers,
            # max_steps=cfg.max_steps,
            precision=cfg.precision,
            strategy=strategy,
            sync_batchnorm=True,
            # use_distributed_sampler=False,
            max_epochs=cfg.max_epochs,
            overfit_batches=overfit,
            # val_check_interval=cfg.val_check_interval,
        )

    trainer.fit(model, data_module, ckpt_path=resume_path)


if __name__ == "__main__":
    main()
