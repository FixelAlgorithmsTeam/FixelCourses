

from typing import Dict

import numpy as np

import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

import pytorch_lightning as pl
from torchmetrics.detection.mean_ap import MeanAveragePrecision

# Royi: For Enums used for models
from enum import auto, Enum, unique


@unique
class OptimizerType(Enum):
  ADAM  = auto()
  SGD   = auto()

@unique
class SchedulerType(Enum):
  LINEAR_LR   = auto()
  CONSTANT_LR = auto()
  ONE_CYCLER  = auto()
  STEP_LR     = auto()


class LitModel(pl.LightningModule):
    def __init__(self, modelNet: nn.Module, dOptConf: Dict = {}, dSchedConf: Dict = {}):
        super().__init__()
        #TODO: Export the net to TensorBoard

        # self.save_hyperparameters() #<! Cause issues with loading the data

        # self.example_input_array = [torch.rand(3, 100, 100), torch.rand(3, 200, 200)]
        self.example_input_array = [[torch.rand(3, 100, 100)]] #<! List in a list since a list is splatted
        
        trainMapScore = MeanAveragePrecision(box_format = 'xyxy', iou_type = 'bbox')
        valMapScore   = MeanAveragePrecision(box_format = 'xyxy', iou_type = 'bbox')
        
        self.modelNet       = modelNet

        self.trainMapScore  = trainMapScore
        self.valMapScore    = valMapScore

        self.dOptConf   = dOptConf.copy() #<! Prevent `pop` to affect to original data
        self.dSchedConf = dSchedConf.copy() #<! Prevent `pop` to affect to original data

        self.lastEpoch = -1

    def forward(self, x):
        return self.modelNet(x)

    def training_step(self, batch, batch_idx):
        lImg, lTarget = batch
        
        dLoss   = self.modelNet(lImg, lTarget) #<! Loss is built in the model
        lossVal = sum(loss for loss in dLoss.values())
        
        lossCls         = dLoss['loss_classifier'].detach()
        lossBoxReg      = dLoss['loss_box_reg'].detach()
        # lossObj         = dLoss['loss_objectness'].detach()
        # lossRpnBoxReg   = dLoss['loss_rpn_box_reg'].detach()

        self.log('train_loss', lossVal, prog_bar = True, on_epoch = True)
        # self.log_dict(dLoss, prog_bar = True, on_epoch = True)
        self.log('train_loss_cls', lossCls, prog_bar = True, on_epoch = True)
        self.log('loss_box_reg', lossBoxReg, prog_bar = True, on_epoch = True)
        
        return lossVal
    
    def validation_step(self, batch, batch_idx):
        # TODO: Export an image to TB
        lImg, lTarget = batch
        lPred = self(lImg)

        self.valMapScore.update(lPred, lTarget)
        dMapScore = self.valMapScore.compute()

        # https://scribe.rip/3f330efe697b
        self.log('map_050', dMapScore['map_50'], prog_bar = True, on_epoch = True)
        self.log('map_075', dMapScore['map_75'], prog_bar = True, on_epoch = True)

        # if self.current_epoch != self.lastEpoch:
        #     figData = ExportNetPred(x[0].clone().detach().cpu().numpy(), y[0].clone().detach().cpu().numpy() , y_hat[0].clone().detach().cpu().numpy())
        #     self.logger.experiment.add_image('images', figData, self.current_epoch, dataformats = 'HWC')
        #     self.lastEpoch = self.current_epoch
    
    def predict_step(self, batch, batch_idx):
        # Assumes Data Loader which is like Training
        # For manual work, do `modelTrainer.lightning_module.forward(x)`

        lImg, lTarget = batch
        return self(lImg)

    def configure_optimizers(self):

        dOptConf    = self.dOptConf
        dSchedConf  = self.dSchedConf

        # Optimizer Configuration
        optimizer_type  = dOptConf.pop('optimizer_type', OptimizerType.ADAM)
        lr              = dOptConf.pop('lr', 0.001)
        if optimizer_type is OptimizerType.ADAM:
            betas           = dOptConf.pop('betas', (0.9, 0.999))
            eps             = dOptConf.pop('eps', 1e-08)
            weight_decay    = dOptConf.pop('weight_decay', 0)
            amsgrad         = dOptConf.pop('amsgrad', False)
            # dOptConf.clear()
            # dOptConf{'lr': lr}
            # dOptConf{'optimizer_type': optimizer_type}
            # dOptConf{'betas': betas}
            # dOptConf{'eps': eps}
            # dOptConf{'weight_decay': weight_decay}
            # dOptConf{'amsgrad': amsgrad}
            modelOpt        = torch.optim.Adam(self.parameters(), lr = lr, betas = betas, eps = eps, weight_decay = weight_decay, amsgrad = amsgrad)
        elif optimizer_type is OptimizerType.SGD:
            momentum        = dOptConf.pop('momentum', 0)
            weight_decay    = dOptConf.pop('weight_decay', 0)
            dampening       = dOptConf.pop('dampening', 0)
            nesterov        = dOptConf.pop('nesterov', False)
            modelOpt        = torch.optim.SGD(self.parameters(), lr = lr, momentum = momentum, weight_decay = weight_decay, dampening = dampening, nesterov = nesterov)
        else:
            raise ValueError(f'The given option for `optimizer_type`: {dOptConf["optimizer_type"]} is invalid')
       
        # Scheduler Configuration
        scheduler_type  = dSchedConf.pop('scheduler_type', SchedulerType.CONSTANT_LR)
        if scheduler_type is SchedulerType.CONSTANT_LR:
            modelScheduler = torch.optim.lr_scheduler.ConstantLR(modelOpt, factor = 1) #<! Constant LR
        elif scheduler_type is SchedulerType.LINEAR_LR:
            start_factor    = dSchedConf.pop('start_factor', 1.0 / 3.0)
            end_factor      = dSchedConf.pop('end_factor', 1)
            total_iters     = dSchedConf.pop('total_iters', 5)
            last_epoch      = dSchedConf.pop('last_epoch', -1)
            modelScheduler  = torch.optim.lr_scheduler.LinearLR(modelOpt, start_factor = start_factor, end_factor = end_factor, total_iters = total_iters, last_epoch = last_epoch)
        elif scheduler_type is SchedulerType.ONE_CYCLER:
            raise ValueError(f'The OneCycleLR requires scheduling at the batch level, not implemented at the moment')
        elif scheduler_type is SchedulerType.STEP_LR:
            step_size       = dSchedConf.pop('step_size', 1)
            gamma           = dSchedConf.pop('gamma', 0.1)
            last_epoch      = dSchedConf.pop('last_epoch', -1)
            modelScheduler  = torch.optim.lr_scheduler.StepLR(modelOpt, step_size = step_size, gamma = gamma, last_epoch = last_epoch)
        else:
            raise ValueError(f'The given option for `scheduler_type`: {dSchedConf["scheduler_type"]} is invalid')
        
        return [modelOpt], [modelScheduler]
    


# Auxiliary Functions

def PlotNetPred(inputImg: np.ndarray, labelsImg: np.ndarray, outImg: np.ndarray, binClassThr = 0.0, hF: plt.Figure = None) -> plt.Figure:

    if (len(outImg.shape) > 2) and (outImg.shape[0] > 1): #<! Data is hot encoded
        # Multi class 
        num_class   = outImg.shape[0]
        outImg      = np.argmax(outImg, axis = 0)
    else:
        # Binary Class
        num_class   = 2 
        outImg      = np.int8(np.squeeze(outImg) > binClassThr)
    
    if hF is None:
        hF, mHA = plt.subplots(nrows = 1, ncols = 3, figsize = (20, 10))
    else:
        mHA = hF.subplots(nrows = 1, ncols = 3, figsize = (20, 10))

    # Input Image
    inputImg = inputImg.transpose((1, 2, 0)) #<! C x H x W -> H x W x C
    mHA[0].imshow(inputImg) #<! RGB Image
    mHA[0].set_title('Input Image')

    # Labels Image
    imgAx = mHA[1].imshow(np.squeeze(labelsImg), cmap = 'jet', interpolation = 'none', vmin = 0, vmax = num_class - 1)
    mHA[1].set_title('Labels Image')

    # Output Image
    imgAx = mHA[2].imshow(outImg, cmap = 'jet', interpolation = 'none', vmin = 0, vmax = num_class - 1)
    mHA[2].set_title('Output Image')

    hF.colorbar(imgAx, ax = mHA[:], orientation = 'horizontal')

    return hF

def ExportNetPred(inputImg: np.ndarray, labelsImg: np.ndarray, outImg: np.ndarray, binClassThr = 0.0) -> np.ndarray:

    if (len(outImg.shape) > 2) and (outImg.shape[0] > 1): #<! Data is hot encoded
        # Multi class 
        num_class   = outImg.shape[0]
        outImg      = np.argmax(outImg, axis = 0)
    else:
        # Binary Class
        num_class   = 2 
        outImg      = np.int8(np.squeeze(outImg) > binClassThr)
    
    hF, mHA = plt.subplots(nrows = 1, ncols = 3, figsize = (20, 10))

    # Input Image
    inputImg = inputImg.transpose((1, 2, 0)) #<! C x H x W -> H x W x C
    mHA[0].imshow(inputImg) #<! RGB Image
    mHA[0].set_title('Input Image')

    # Labels Image
    imgAx = mHA[1].imshow(np.squeeze(labelsImg), cmap = 'jet', interpolation = 'none', vmin = 0, vmax = num_class - 1)
    mHA[1].set_title('Labels Image')

    # Output Image
    imgAx = mHA[2].imshow(outImg, cmap = 'jet', interpolation = 'none', vmin = 0, vmax = num_class - 1)
    mHA[2].set_title('Output Image')

    hF.colorbar(imgAx, ax = mHA[:], orientation = 'horizontal')

    fig_data = ExportFigDataToNumpy(hF)

    return fig_data

def ExportFigDataToNumpy(hF):
  
    hF.canvas.draw()

    mData = np.frombuffer(hF.canvas.tostring_rgb(), dtype = np.uint8)
    num_cols, num_rows = hF.canvas.get_width_height()
    mData = mData.reshape((int(num_rows), int(num_cols), -1))   
    return mData