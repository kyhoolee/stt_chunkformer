from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from .finetune_utils import load_model_only

def build_model_and_optimizer(conf, device, total_steps):
    # load model + tokenizer from your init()
    model, char_dict = load_model_only(conf.model_checkpoint, device)
    tokenizer = conf.tokenizer_from_vocab(conf.vocab_path)

    # optimizer & scheduler
    optimizer = AdamW(model.parameters(), lr=conf.lr, weight_decay=conf.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=conf.lr * 0.01)
    
    return model, tokenizer, optimizer, scheduler
