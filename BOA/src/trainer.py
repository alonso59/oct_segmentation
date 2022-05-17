import torch
from callbacks import TensorboardWriter


def trainer(num_epochs,
            train_loader,
            val_loader,
            model,
            optimizer,
            loss_fn,
            metric,
            device,
            checkpoint_path,
            scheduler,
            name_model,
            callback_stop_value,
            tb_dir,
            logger
            ):
    """ Create log interface """
    writer = TensorboardWriter(metric=metric, name_dir=tb_dir + 'tb_' + name_model + '/')
    iter_num = 0.0
    iter_val = 0.0
    stop_early = 0
    best_valid_loss = float("inf")

    for epoch in range(num_epochs):
        lr_ = optimizer.param_groups[0]["lr"]
        str = f"Epoch: {epoch+1}/{num_epochs} --loss_fn:{loss_fn.__name__} --model:{name_model} --lr:{lr_:.4e}"
        logger.info(str)
        train_loss, train_metric, iter_num, lr_ = train(
            loader=train_loader,
            model=model,
            writer=writer,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=device,
            metric=metric,
            iter_num=iter_num,
        )
        val_loss, val_metric, iter_val = validation(
            model, val_loader, loss_fn, metric, device, iter_val, writer, iter_plot_img)

        scheduler.step()

        writer.learning_rate(optimizer.param_groups[0]["lr"], epoch)

        """ Saving the model """
        if val_loss < best_valid_loss:
            str_print = f"Valid loss improved from {best_valid_loss:2.4f} to {val_loss:2.4f}. Saving checkpoint: {checkpoint_path}"
            best_valid_loss = val_loss
            stop_early = 0
        else:
            stop_early += 1
            str_print = f"Valid loss not improved: {best_valid_loss:2.4f}, ESC: {stop_early}/{callback_stop_value}"
        if stop_early == callback_stop_value:
            logger.info('+++++++++++++++++ Stop training early +++++++++++++')
            break
        logger.info(f'----> Train {metric.__name__}: {train_metric:.4f} \t Val. {metric.__name__}: {val_metric:.4f}')
        logger.info(f'----> Train Loss: {train_loss:.4f} \t Val. Loss: {val_loss:.4f}')
        logger.info(str_print)


def train(loader, model, writer, optimizer, loss_fn, device, metric, iter_num):
    train_loss = 0.0
    train_iou = 0.0
    model.train()
    for batch_idx, (x, y) in enumerate(loader):
        x = x.type(torch.float).to(device)
        y = y.type(torch.long).to(device)
        # forward
        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # metrics
        m0 = metric(y_pred, y)
        train_iou += m0.mean()
        train_loss += loss.item()
        # tensorboard callbacks
        writer.per_iter(loss.item(), m0.mean(), iter_num, name='Train')
        iter_num = iter_num + 1
    return train_loss / len(loader), train_iou / len(loader), iter_num, optimizer.param_groups[0]["lr"]


def validation(model, loader, loss_fn, metric, device, iter_val, writer, iter_plot_img):
    valid_loss = 0.0
    valid_iou = 0.0
    model.eval()
    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(loader):
            x = x.type(torch.float).to(device)
            y = y.type(torch.long).to(device)
            y_pred = model(x)
            loss = loss_fn(y_pred, y)
            m0 = metric(y_pred, y)
            # accumulate metrics and loss items
            valid_iou += m0.mean()
            valid_loss += loss.item()
            # tensorboard callbacks
            writer.per_iter(loss.item(), m0.mean(), iter_val, name='Val')
    return valid_loss / len(loader), valid_iou / len(loader), iter_val


def eval(model, loader, loss_fn, device):
    eval_loss = 0.0
    model.eval()
    for batch_idx, (x, y) in enumerate(loader):
        x = x.type(torch.float).to(device)
        y = y.type(torch.long).to(device)
        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        eval_loss += loss.item()
    return eval_loss / len(loader)
