def update_lr(train_loss_history_var, config_var, train_loss_var):
    return (len(train_loss_history_var) > 1) and (train_loss_history_var[-2] * config_var.lr_decay_threshold < train_loss_var)