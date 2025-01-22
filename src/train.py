import logging
import os
from metrics_visibility_fast import AverageMeter, average_mAP, NMS
import time
from tqdm import tqdm
import torch
import numpy as np
import math
from preprocessing import batch2long, timestamps2long
from json_io import predictions2json
from SoccerNet.Downloader import getListGames
import random
import matplotlib.pyplot as plt 
from torch.optim.lr_scheduler import LambdaLR


# Set random seed for reproducibility
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# Utility for plotting loss curves
def plot_loss_curves(train_losses, val_losses, model_name):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Training Loss")
    plt.plot(val_losses, label="Validation Loss", linestyle='--')
    plt.title(f"Loss Curves: {model_name}")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    os.makedirs(os.path.join("models", model_name), exist_ok=True)
    plt.savefig(os.path.join("models", model_name, "loss_curves.png"))
    plt.show()


# Utility for plotting learning rate schedule
def plot_lr_scheduler(lr_rates, model_name):
    plt.figure(figsize=(10, 5))
    plt.plot(lr_rates, label="Learning Rate")
    plt.title(f"Learning Rate Schedule: {model_name}")
    plt.xlabel("Epochs")
    plt.ylabel("Learning Rate")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join("models", model_name, "lr_schedule.png"))
    plt.show()

# Define the warmup scheduler
def warmup_scheduler(optimizer, warmup_epochs, base_lr):
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            # Gradually increase the learning rate
            return (epoch + 1) / warmup_epochs
        else:
            # Use the base learning rate
            return 1.0
    return LambdaLR(optimizer, lr_lambda=lr_lambda)

def trainer(train_loader,
            val_loader,
            val_metric_loader,
            test_loader,
            model,
            optimizer,
            scheduler,
            criterion,
            weights,
            model_name,
            max_epochs=1000,
            warmup_epochs=0,  # Add warmup_epochs parameter
            evaluation_frequency=20):

    logging.info("Start training with warmup")

    # Initialize warmup scheduler
    warmup_lr_scheduler = warmup_scheduler(optimizer, warmup_epochs, optimizer.param_groups[0]['lr'])
    best_loss = 9e99
    best_metric = -1

    train_losses = []
    val_losses = []
    lr_rates = []

    for epoch in range(max_epochs):
        best_model_path = os.path.join("models", model_name, "model.pth.tar")

        # Train for one epoch
        loss_training = train(
            train_loader,
            model,
            criterion,
            weights,
            optimizer,
            epoch + 1,
            train=True)
        train_losses.append(loss_training)
        lr_rates.append(optimizer.param_groups[0]['lr'])

        # Evaluate on validation set
        # if (epoch + 1) % evaluation_frequency == 0 or epoch == max_epochs - 1:
        loss_validation = train(
            val_loader,
            model,
            criterion,
            weights,
            optimizer,
            epoch + 1,
            train=False)
        val_losses.append(loss_validation)
        logging.info(f"Epoch [{epoch + 1}/{max_epochs}] Validation Loss: {loss_validation:.4f}")

        state = {
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_loss': best_loss,
            'optimizer': optimizer.state_dict(),
        }
        os.makedirs(os.path.join("models", model_name), exist_ok=True)

        # Remember best loss and save checkpoint
        is_better = loss_validation < best_loss
        best_loss = min(loss_validation, best_loss)

        # Save the best model based on loss only if the evaluation frequency is too long
        if is_better and evaluation_frequency > max_epochs:
            torch.save(state, best_model_path)

        # Test the model on the validation set
        if epoch % evaluation_frequency == 0 and epoch != 0:
            performance_validation = test(
                val_metric_loader,
                model,
                model_name)

            performance_validation = performance_validation[0]
            logging.info("Validation performance at epoch " + str(epoch + 1) + " -> " + str(performance_validation))

            is_better_metric = performance_validation > best_metric
            best_metric = max(performance_validation, best_metric)

            # Save the best model based on metric only if the evaluation frequency is short enough
            if is_better_metric and evaluation_frequency <= max_epochs:
                torch.save(state, best_model_path)
                performance_test = test(
                    test_loader,
                    model,
                    model_name, save_predictions=True)
                performance_test = performance_test[0]

                logging.info("Test performance at epoch " + str(epoch + 1) + " -> " + str(performance_test))

        # Learning rate scheduler update
        prevLR = optimizer.param_groups[0]['lr']
        # Apply warmup scheduler during the warmup phase
        if epoch < warmup_epochs:
            warmup_lr_scheduler.step()
        else:
            # After warmup, use the main scheduler
            prevLR = optimizer.param_groups[0]['lr']
            scheduler.step(loss_validation)  # Using training loss for scheduler update
            currLR = optimizer.param_groups[0]['lr']
            if currLR != prevLR and scheduler.num_bad_epochs == 0:
                logging.info("Plateau Reached!")

        if prevLR < 2 * scheduler.eps and scheduler.num_bad_epochs >= scheduler.patience:
            logging.info("Plateau Reached and no more reduction -> Exiting Loop")
            break

    # Plot final loss curves at the end of training
    plot_loss_curves(train_losses, val_losses, model_name)
    plot_lr_scheduler(lr_rates, model_name)

    return

def train(dataloader, model, criterion, weights, optimizer, epoch, train=False):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    losses_segmentation = AverageMeter()
    losses_spotting = AverageMeter()

    model.train() if train else model.eval()

    end = time.time()
    with tqdm(enumerate(dataloader), total=len(dataloader), ncols=160, desc=f"Epoch {epoch}") as t:
        for i, (feats, labels, targets) in t:
            data_time.update(time.time() - end)

            feats = feats.cuda().unsqueeze(1)
            labels = labels.cuda().float()
            targets = targets.cuda().float()

            output_segmentation, output_spotting = model(feats)
            loss_segmentation = criterion[0](labels, output_segmentation)
            loss_spotting = criterion[1](targets, output_spotting)
            loss = weights[0] * loss_segmentation + weights[1] * loss_spotting

            losses.update(loss.item(), feats.size(0))
            losses_segmentation.update(loss_segmentation.item(), feats.size(0))
            losses_spotting.update(loss_spotting.item(), feats.size(0))

            if train:
                optimizer.zero_grad()
                loss.backward()

                # Gradient clipping
                # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                optimizer.step()

            batch_time.update(time.time() - end)
            end = time.time()

            t.set_postfix({
                "Loss": f"{losses.avg:.4e}",
                "Loss Seg": f"{losses_segmentation.avg:.4e}",
                "Loss Spot": f"{losses_spotting.avg:.4e}",
            })

    return losses.avg



def test(dataloader,model, model_name, save_predictions=False):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    spotting_grountruth = list()
    spotting_grountruth_visibility = list()
    spotting_predictions = list()
    segmentation_predictions = list()

    chunk_size = model.chunk_size
    receptive_field = model.receptive_field

    model.eval()

    end = time.time()
    with tqdm(enumerate(dataloader), total=len(dataloader), ncols=120) as t:
        for i, (feat_half1, feat_half2, label_half1, label_half2) in t:
            data_time.update(time.time() - end)

            feat_half1 = feat_half1.cuda().squeeze(0)
            label_half1 = label_half1.float().squeeze(0)
            feat_half2 = feat_half2.cuda().squeeze(0)
            label_half2 = label_half2.float().squeeze(0)


            feat_half1=feat_half1.unsqueeze(1)
            feat_half2=feat_half2.unsqueeze(1)

            # Compute the output
            output_segmentation_half_1, output_spotting_half_1 = model(feat_half1)
            output_segmentation_half_2, output_spotting_half_2 = model(feat_half2)


            timestamp_long_half_1 = timestamps2long(output_spotting_half_1.cpu().detach(), label_half1.size()[0], chunk_size, receptive_field)
            timestamp_long_half_2 = timestamps2long(output_spotting_half_2.cpu().detach(), label_half2.size()[0], chunk_size, receptive_field)
            segmentation_long_half_1 = batch2long(output_segmentation_half_1.cpu().detach(), label_half1.size()[0], chunk_size, receptive_field)
            segmentation_long_half_2 = batch2long(output_segmentation_half_2.cpu().detach(), label_half2.size()[0], chunk_size, receptive_field)

            spotting_grountruth.append(torch.abs(label_half1))
            spotting_grountruth.append(torch.abs(label_half2))
            spotting_grountruth_visibility.append(label_half1)
            spotting_grountruth_visibility.append(label_half2)
            spotting_predictions.append(timestamp_long_half_1)
            spotting_predictions.append(timestamp_long_half_2)
            segmentation_predictions.append(segmentation_long_half_1)
            segmentation_predictions.append(segmentation_long_half_2)


    # Transformation to numpy for evaluation
    targets_numpy = list()
    closests_numpy = list()
    detections_numpy = list()
    for target, detection in zip(spotting_grountruth_visibility,spotting_predictions):
        target_numpy = target.numpy()
        targets_numpy.append(target_numpy)
        detections_numpy.append(NMS(detection.numpy(), 20*model.framerate))
        closest_numpy = np.zeros(target_numpy.shape)-1
        #Get the closest action index
        for c in np.arange(target_numpy.shape[-1]):
            indexes = np.where(target_numpy[:,c] != 0)[0].tolist()
            if len(indexes) == 0 :
                continue
            indexes.insert(0,-indexes[0])
            indexes.append(2*closest_numpy.shape[0])
            for i in np.arange(len(indexes)-2)+1:
                start = max(0,(indexes[i-1]+indexes[i])//2)
                stop = min(closest_numpy.shape[0], (indexes[i]+indexes[i+1])//2)
                closest_numpy[start:stop,c] = target_numpy[indexes[i],c]
        closests_numpy.append(closest_numpy)

    # Save the predictions to the json format
    if save_predictions:
        list_game = getListGames(dataloader.dataset.split)
        for index in np.arange(len(list_game)):
            predictions2json(detections_numpy[index*2], detections_numpy[(index*2)+1],"outputs/", list_game[index], model.framerate)


    # Compute the performances
    a_mAP, a_mAP_per_class, a_mAP_visible, a_mAP_per_class_visible, a_mAP_unshown, a_mAP_per_class_unshown = average_mAP(targets_numpy, detections_numpy, closests_numpy, model.framerate)
    
    print("Average mAP: ", a_mAP)
    print("Average mAP visible: ", a_mAP_visible)
    print("Average mAP unshown: ", a_mAP_unshown)
    print("Average mAP per class: ", a_mAP_per_class)
    print("Average mAP visible per class: ", a_mAP_per_class_visible)
    print("Average mAP unshown per class: ", a_mAP_per_class_unshown)

    return a_mAP, a_mAP_per_class, a_mAP_visible, a_mAP_per_class_visible, a_mAP_unshown, a_mAP_per_class_unshown