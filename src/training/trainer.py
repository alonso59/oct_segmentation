import torch
from tqdm import tqdm
from .callbacks import TensorboardWriter
from .metrics import mIoU

def trainer(num_epochs, train_loader, val_loader, model, optimizer, loss_fn, metric, device, checkpoint_path, scheduler,
            iter_plot_img, name_model, callback_stop_value, tb_dir, logger):
    """ Create log interface """
    writer = TensorboardWriter(metric=metric, name_dir=tb_dir + 'tb_' + name_model + '/')
    iter_train = 0.0
    iter_val = 0.0
    stop_early = 0
    best_valid_loss = float("inf")

    for epoch in range(num_epochs):
        lr_ = optimizer.param_groups[0]["lr"]
        str = f"Epoch: {epoch+1}/{num_epochs} --loss_fn:{loss_fn.__name__} --model:{name_model} --lr:{lr_:.4e}"
        logger.info(str)

        train_loss, train_iou, iter_train = train(
            loader=train_loader, model=model, writer=writer, optimizer=optimizer, loss_fn=loss_fn, device=device,
            metric=metric, iter_train=iter_train)
        val_loss, val_iou, iter_val = validation(
            model, val_loader, loss_fn, metric, device, iter_val, writer, iter_plot_img)

        """ scheduler learning rate """
        scheduler.step()
        # metric_names = ['Pixel Acc', 'Dice', 'Precision', 'Recall']
        writer.learning_rate(optimizer.param_groups[0]["lr"], epoch)
        writer.loss_epoch(train_loss, val_loss, epoch)
        writer.metrics_epoch(train_metric=train_iou, val_metric=val_iou, step=epoch, metric_name='mIoU')

        """ Saving the model """
        if val_loss < best_valid_loss:
            str_print = f"Valid loss improved from {best_valid_loss:2.4f} to {val_loss:2.4f}. Saving checkpoint: {checkpoint_path}"
            best_valid_loss = val_loss
            torch.save(model, checkpoint_path + f'/model.pth')
            torch.save(model.state_dict(), checkpoint_path + f'/weights.pth')
            stop_early = 0
        else:
            stop_early += 1
            str_print = f"Valid loss not improved: {best_valid_loss:2.4f}, ESC: {stop_early}/{callback_stop_value}"
        if stop_early == callback_stop_value:
            logger.info('+++++++++++++++++ Stop training early +++++++++++++')
            break
        logger.info(f'----> Train Loss: {train_loss:.5f} \t Val. Loss: {val_loss:.5f}')
        logger.info(f'----> Train mIoU: {train_iou:.5f} \t Val. mIoU: {val_iou:0.5f}')
        logger.info(str_print)
    torch.save(model, checkpoint_path + f'/model_last.pth')
    torch.save(model.state_dict(), checkpoint_path + f'/weights_last.pth')


def train(loader, model, writer, optimizer, loss_fn, device, metric, iter_train):
    train_loss = 0.0
    train_metric = 0.0
    loop = tqdm(loader, ncols=150)
    model.train()
    mean_iou = mIoU(device)
    for batch_idx, (x, y) in enumerate(loop):
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
        iou = mean_iou(y_pred, y)
        metric_value = metric(y, y_pred)
        train_metric += iou.mean().item()
        train_loss += loss.item()
        # update tqdm loop
        loop.set_postfix(
            Pixel_acc=metric_value[0],
            Dice=metric_value[1],
            Precision=metric_value[2],
            Recall=metric_value[3],
            Loss=loss.item(),
            mIoU=iou.mean().item()
            )
        metric_value.append(iou.mean().item())
        # tensorboard callbacks
        writer.loss_iter(loss.item(), iter_train, stage='Train')
        metric_names = ['Pixel Acc', 'Dice', 'Precision', 'Recall', 'mIoU']
        for i, names in enumerate(metric_names):
            writer.metric_iter(
                metric_value[i],
                iter_train, stage='Train', metric_name=names)
        if iter_train % (len(loader) * 10) == 0:
            print('\nSaving examples in Tensorboard...')
            writer.save_images(x, y, y_pred, iter_train, device, tag='train')
        iter_train = iter_train + 1
    return train_loss / len(loader), train_metric / len(loader), iter_train


def validation(model, loader, loss_fn, metric, device, iter_val, writer, iter_plot_img):
    valid_loss = 0.0
    valid_metric = 0.0
    loop = tqdm(loader, ncols=150)
    model.eval()
    mean_iou = mIoU(device)
    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(loop):
            x = x.type(torch.float).to(device)
            y = y.type(torch.long).to(device)
            y_pred = model(x)
            loss = loss_fn(y_pred, y)
            metric_value = metric(y, y_pred)
            iou = mean_iou(y_pred, y)
            # accumulate metrics and loss items
            valid_metric += iou.mean().item()
            valid_loss += loss.item()
            # update tqdm loop
            loop.set_postfix(
                Pixel_acc=metric_value[0],
                Dice=metric_value[1],
                Precision=metric_value[2],
                Recall=metric_value[3],
                Loss=loss.item(),
                mIoU=iou.mean().item()
                )
            metric_value.append(iou.mean().item())
            # tensorboard callbacks
            writer.loss_iter(loss.item(), iter_val, stage='Val')
            metric_names = ['Pixel Acc', 'Dice', 'Precision', 'Recall', 'mIoU']
            for i, names in enumerate(metric_names):
                writer.metric_iter(
                    metric_value[i],
                    iter_val, stage='Val', metric_name=names)

            if iter_val % (len(loader) * iter_plot_img) == 0:
                print('\nSaving examples in Tensorboard...')
                writer.save_images(x, y, y_pred, iter_val, device, tag='val')
            iter_val += 1
    return valid_loss / len(loader), valid_metric / len(loader), iter_val


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
