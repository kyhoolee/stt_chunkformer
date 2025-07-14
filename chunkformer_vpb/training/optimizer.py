from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from .finetune_config import FinetuneConfig
from .finetune_utils import load_model_only

def build_model_and_optimizer(conf: FinetuneConfig, device, total_steps):
    # load model + tokenizer from your init()
    model, _ = load_model_only(conf.model.checkpoint, device)
    tokenizer = conf.tokenizer.tokenizer

    # optimizer & scheduler
    # optimizer = AdamW(model.parameters(), lr=conf.lr, weight_decay=conf.weight_decay)
    # scheduler = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=conf.lr * 0.01)

    optimizer = AdamW(model.parameters(), lr=conf.training.lr, weight_decay=conf.training.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=conf.training.lr*0.01)

    return model, tokenizer, optimizer, scheduler
