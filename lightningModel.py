"""
Inria Chile
Support stuff for training image-based classifiers using Pytorch Lightning.
"""

import logging
from abc import ABC, abstractmethod
from typing import Callable, List, Optional, Union
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import torch.nn.functional as F
from torch import Tensor

import torch
import torch.nn as nn
from pytorch_lightning import LightningModule, Callback
from torch.autograd import Variable

import torchvision.models as models

import flash
from flash.image import ImageClassifier

from torch.optim.lr_scheduler import ReduceLROnPlateau

#from focal_loss.focal_loss import FocalLoss

from torchmetrics import MetricCollection
from torchmetrics.classification import MulticlassF1Score, MulticlassAccuracy, MulticlassPrecision, MulticlassRecall, BinaryAccuracy, BinaryPrecision, BinaryRecall, BinaryF1Score


logger = logging.getLogger(__file__)

class FocalLoss(nn.Module):
    """
        This code was originally taken from the link below, but has been updated.
        This code is from Pytorch Lightning ArcFace Focal Loss: https://www.kaggle.com/code/zeta1996/pytorch-lightning-arcface-focal-loss
        This criterion is a implemenation of Focal Loss, which is proposed in 
        Focal Loss for Dense Object Detection.

            Loss(x, class) = - \alpha (1-softmax(x)[class])^gamma \log(softmax(x)[class])

        The losses are averaged across observations for each minibatch.

        Args:
            alpha(1D Tensor, Variable) : the scalar factor for this criterion
            gamma(float, double) : gamma > 0; reduces the relative loss for well-classiﬁed examples (p > .5), 
                                   putting more focus on hard, misclassiﬁed examples
            size_average(bool): By default, the losses are averaged over observations for each minibatch.
                                However, if the field size_average is set to False, the losses are
                                instead summed for each minibatch.


    """
    def __init__(self, class_num, alpha=None, gamma=2, size_average=True):
        super(FocalLoss, self).__init__()
        if alpha == None:
            self.alpha = Variable(torch.ones(class_num, 1))
        else:
            if isinstance(alpha, Variable):
                self.alpha = alpha
            else:
                self.alpha = Variable(alpha)
        self.gamma = gamma
        self.class_num = class_num
        self.size_average = size_average

    def forward(self, inputs, targets):
        N = inputs.size(0)
        C = inputs.size(1)
        P = F.softmax(inputs, dim=1)

        class_mask = torch.zeros((N, C), device=inputs.device)
        class_mask.scatter_(1, targets.view(-1, 1), 1.)

        if inputs.is_cuda and not self.alpha.is_cuda:
            self.alpha = self.alpha.to(inputs.device)
        alpha = self.alpha[targets.view(-1)]

        probs = (P * class_mask).sum(1).view(-1, 1)
        log_p = probs.log()

        batch_loss = -alpha * (torch.pow((1 - probs), self.gamma)) * log_p

        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss

class classifier_model(LightningModule, ABC):
    def __init__(   self,
                    results: Path,
                    model_name: str,
                    weights: Tensor, 
                    num_classes: int,
                    pretrained: Optional[bool] = False,
                    fine_tune: Optional[bool] = False,
                    learning_rate: float = 1e-03,
                    dropout: float = 0,
                    scheduler: str = None,
                    scheduler_parameters: Optional[dict] = None,
                    loss_function: str = None,
                    
                ):
        super().__init__()


        # log hyperparameters
        self.save_hyperparameters() # ignore=[Model_path]
        self.RESULTS = results
        self.model_name = model_name
        self.learning_rate = learning_rate
        self.scheduler = scheduler
        self.scheduler_parameters = scheduler_parameters

        # defining loss
        if loss_function == 'focal_loss':
            self.criterion = FocalLoss(class_num=num_classes, alpha = weights)
        else:
            self.criterion = nn.CrossEntropyLoss()
        # create model
        self.feature_extractor, self.head = self.get_model_and_feature_extractor(model_name, pretrained, num_classes)
        #self.flatten = nn.Flatten(start_dim=1)


        # Define metrics
        class_metrics = MetricCollection([
                                            MulticlassF1Score(num_classes=self.hparams.num_classes, average='macro'),
                                            MulticlassAccuracy(num_classes=self.hparams.num_classes, average='weighted'),
                                            MulticlassPrecision(num_classes=self.hparams.num_classes, average='macro'),
                                            MulticlassRecall(num_classes=self.hparams.num_classes, average='macro'),
                                        ])
        self.train_metrics_class = class_metrics.clone(prefix='train_')
        self.val_metrics_class = class_metrics.clone(prefix='val_')
        self.test_metrics_class = class_metrics.clone(prefix='test_')
        
        if fine_tune:
            for param in self.feature_extractor.parameters():
                param.requires_grad = False
        
        

    
    def get_model_and_feature_extractor(self, model_name, pretrained, num_classes):
        model = ImageClassifier(backbone=model_name, num_classes=num_classes, pretrained=pretrained).adapter
        feature_extractor = model.backbone
        head = model.head
        return feature_extractor, head
       
    def forward(self, x):
        embedding = self.feature_extractor(x)
        output = self.head(embedding)
        return output, embedding

    def training_step(self, batch, batch_idx):

        # Get targets 
        images, class_name_target = batch

        # Get predictions
        class_preds, _ = self.forward(images)

        # Calculate loss
        loss = self.criterion(class_preds, class_name_target)

        # Update metrics
        self.train_metrics_class(class_preds, class_name_target)
                
        self.log('train_loss', loss, sync_dist=True)
        
        return {"loss": loss}


    def training_epoch_end(self, outputs): 
        # calculate and log the metrics for classification
        train_f1 = self.train_metrics_class['MulticlassF1Score']
        self.log('train_f1', train_f1, on_step=False, on_epoch=True, sync_dist=True)
        train_accuracy = self.train_metrics_class['MulticlassAccuracy']
        self.log('train_accuracy', train_accuracy, on_step=False, on_epoch=True, sync_dist=True)
        train_precision = self.train_metrics_class['MulticlassPrecision']
        self.log('train_precision', train_precision, on_step=False, on_epoch=True, sync_dist=True)
        train_recall = self.train_metrics_class['MulticlassRecall']
        self.log('train_recall', train_recall, on_step=False, on_epoch=True, sync_dist=True)

    def validation_step(self, batch, batch_idx):

        # Get targets 
        images, class_name_target = batch

        # Get predictions
        class_preds, embedding = self.forward(images)

        # Calculate loss
        loss = self.criterion(class_preds, class_name_target)

        # Update metrics
        self.val_metrics_class(class_preds, class_name_target)
                
        self.log('val_loss', loss, sync_dist=True)
        
        if self.current_epoch == 1:
            return {"val_loss": loss,
                "c_target": class_name_target.to('cpu'),
                "embedding": embedding.to('cpu')}
        else:
            return {"val_loss": loss}


    def validation_epoch_end(self, outputs): 
        # calculate and log the metrics for classification
        val_f1 = self.val_metrics_class['MulticlassF1Score']
        self.log('val_f1', val_f1, on_step=False, on_epoch=True, sync_dist=True)
        val_accuracy = self.val_metrics_class['MulticlassAccuracy']
        self.log('val_accuracy', val_accuracy, on_step=False, on_epoch=True, sync_dist=True)
        val_precision = self.val_metrics_class['MulticlassPrecision']
        self.log('val_precision', val_precision, on_step=False, on_epoch=True, sync_dist=True)
        val_recall = self.val_metrics_class['MulticlassRecall']
        self.log('val_recall', val_recall, on_step=False, on_epoch=True, sync_dist=True)

        if self.current_epoch == 1:
            c_true = torch.cat([x['c_target'] for x in outputs]).to('cpu')
            TRUE_FILE = Path(self.RESULTS) / 'c_true_first_epoch.pt' 
            embedding = torch.cat([x['embedding'] for x in outputs]).to('cpu')
            EMBEDDING_FILE = Path(self.RESULTS) / 'embedding_first_epoch.pt'

            torch.save(c_true, TRUE_FILE)
            torch.save(embedding, EMBEDDING_FILE)

    def test_step(self, batch, batch_idx):

        # Get targets 
        images, class_name_target = batch

        # Get predictions
        class_preds, embedding = self.forward(images)

        # Calculate loss
        loss = self.criterion(class_preds, class_name_target)

        # Update metrics
        self.test_metrics_class(class_preds, class_name_target)
                
        self.log('test_loss', loss, sync_dist=True)
        
        return {"test_loss": loss,
                "c_target": class_name_target.to('cpu'),
                "c_preds": class_preds.to('cpu'),
                "embedding": embedding.to('cpu')}
                

    def test_epoch_end(self, outputs): 
        # calculate and log the metrics for classification
        test_f1 = self.test_metrics_class['MulticlassF1Score']
        self.log('test_f1', test_f1, on_step=False, on_epoch=True, sync_dist=True)
        test_accuracy = self.test_metrics_class['MulticlassAccuracy']
        self.log('test_accuracy', test_accuracy, on_step=False, on_epoch=True, sync_dist=True)
        test_precision = self.test_metrics_class['MulticlassPrecision']
        self.log('test_precision', test_precision, on_step=False, on_epoch=True, sync_dist=True)
        test_recall = self.test_metrics_class['MulticlassRecall']
        self.log('test_recall', test_recall, on_step=False, on_epoch=True, sync_dist=True)

        c_true = torch.cat([x['c_target'] for x in outputs]).to('cpu')
        TRUE_FILE = Path(self.RESULTS) / f'c_true_test.pt' 
        c_pred = torch.cat([x['c_preds'].argmax(dim=1) for x in outputs]).to('cpu')
        PRED_FILE = Path(self.RESULTS) / f'c_pred_test.pt'
        embedding = torch.cat([x['embedding'] for x in outputs]).to('cpu')
        EMBEDDING_FILE = Path(self.RESULTS) / f'embedding_test.pt'

        torch.save(c_true, TRUE_FILE)
        torch.save(c_pred, PRED_FILE)
        torch.save(embedding, EMBEDDING_FILE)

    def configure_optimizers(self):
        print('#### Configuring Optimizer ####')
        optimizer = torch.optim.AdamW(params=self.parameters(), lr = self.learning_rate)
        print(f'Using a learning rate of {self.learning_rate}')
        
        if self.scheduler == 'cosine_annealing':
            print('#### Using Cosine Annealing Scheduler ####')
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                T_max=self.scheduler_parameters['T_max'],
                                                eta_min=self.scheduler_parameters['eta_min'],
                                                                  verbose = True)
            return {"optimizer": optimizer, "lr_scheduler": scheduler}
        
        elif self.scheduler == 'reduce_lr_on_plateau':
            print('#### Using Reduce On Plateau Scheduler ####')
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                                   'max',
                                                                   factor=0.5,
                                                            patience=self.scheduler_parameters['patience'],
                                                                  min_lr=self.scheduler_parameters['min_lr'],
                                                                  verbose = True)
            return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_f1"}
        
        elif self.scheduler == 'exponential_lr':
            print('#### Using Exponential Scheduler ####')
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer,
                                                                  gamma=self.scheduler_parameters['gamma'],
                                                                  verbose = True)
            return {"optimizer": optimizer, "lr_scheduler": scheduler}
        
        else:
            return optimizer
        
    def update_learning_rate(self, new_learning_rate):
        self.learning_rate = new_learning_rate
        for param_group in self.optimizers:
            param_group['lr'] = new_learning_rate

class BinaryHead(nn.Module, ABC):
    def __init__(   self,
                    feature_extractor_path: Path, 
                    freeze_feature_extractor: Optional[bool] = True,
                    freeze_binary_head: Optional[bool] = False,
                    
                ):
        super().__init__()


        # log hyperparameters
        # create model
        self.embedding_size = 768
        self.feature_extractor, self.binary_head = classifier_model.load_from_checkpoint(Path(__file__).parent /feature_extractor_path).feature_extractor, nn.Linear(self.embedding_size, 2)
        self.softmax = nn.Softmax(dim=-1)
        self.avgpool = nn.AvgPool2d(kernel_size=(7, 7))
        
        if freeze_feature_extractor:
            self._freeze_feature_extractor()
        else:
            self._unfreeze_feature_extractor()
        
        if freeze_binary_head:
            self._freeze_binary_head()
        else:
            self._unfreeze_binary_head()

    def _freeze_feature_extractor(self):
        print('"""Freezing Feature Extractor"""')
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

    def _freeze_binary_head(self):
        print('"""Freezing Binary Head"""')
        for param in self.binary_head.parameters():
            param.requires_grad = False

    def _unfreeze_feature_extractor(self):
        print('"""Freezing Feature Extractor"""')
        for param in self.feature_extractor.parameters():
            param.requires_grad = True

    def _unfreeze_binary_head(self):
        print('"""Freezing Binary Head"""')
        for param in self.binary_head.parameters():
            param.requires_grad = True
       
    def forward(self, x):
        B = x.shape[0]
        patch = self.feature_extractor.patch_embed(x)
        pos = self.feature_extractor.pos_drop(patch)
        features = self.feature_extractor.layers(pos)
        features2 = self.feature_extractor.norm(features)
        embedding = self.avgpool(features2.permute(0, 2, 1).reshape((B, 768, 7, 7))).squeeze()
        output = self.binary_head(embedding)
        return self.softmax(output), features2

class BranchLeaf(nn.Module, ABC):
    def __init__(self, attention_size, num_classes):
        super().__init__()
        
        self.layer3 = nn.Linear(attention_size*2, int(num_classes))


        self.attention1 = nn.Linear(attention_size, 1)
        self.attention2 = nn.Linear(attention_size, 1)
        self.attention3 = nn.Linear(attention_size, 1)
        self.attention4 = nn.Linear(attention_size, 1)

        self.softmax = nn.Softmax(dim=1)


    def forward(self, x_main, x_family, x_other_big, x_other_small):
        weights = torch.unsqueeze(self.softmax(torch.cat([self.attention1(x_main), self.attention2(x_family), self.attention3(x_other_big), self.attention4(x_other_small)], dim=1)), dim=-1)

        inputs = torch.cat([torch.unsqueeze(x_main, dim=-1),torch.unsqueeze(x_family, dim=-1), torch.unsqueeze(x_other_big, dim=-1), torch.unsqueeze(x_other_small, dim=-1)], dim=-1)
        
        attention_output = torch.squeeze(torch.bmm(inputs, weights))
        return self.layer3(torch.concat([x_main, attention_output], dim=1))

class BranchBody(nn.Module, ABC):
    def __init__(self, feature_extractor_path, attention_size):
        super().__init__()
        self.embedding_size = 768
        print(f"feature extrcator path is: {feature_extractor_path}")

        model = classifier_model.load_from_checkpoint(Path(__file__).parent/feature_extractor_path).feature_extractor
        self.feature_block1 = model.layers[3].blocks[0]
        self.feature_block2 = model.layers[3].blocks[1]

        self.branch1 = nn.Sequential(nn.Linear(self.embedding_size, (self.embedding_size + attention_size) //2), nn.Linear((self.embedding_size + attention_size) //2, attention_size))
        self.branch2 = nn.Sequential(nn.Linear(self.embedding_size, (self.embedding_size + attention_size) //2), nn.Linear((self.embedding_size + attention_size) //2, attention_size))
        
        self.avgpool = nn.AvgPool2d(kernel_size=(7, 7))


    def forward(self, x):
        B = x.shape[0]
        features1 = self.feature_block1(x)
        features2 = self.feature_block2(features1)
        f1 = self.avgpool(features1.permute(0, 2, 1).reshape((B, 768, 7, 7))).squeeze()
        f2 = self.avgpool(features2.permute(0, 2, 1).reshape((B, 768, 7, 7))).squeeze()
        return self.branch1(f1), self.branch2(f2)

class Modified_BA_CNN(nn.Module, ABC):
    def __init__(self,
                 feature_extractor_path: Path,
                 num_classes: list[int],
                 attention_size= int,
                 freeze_feature_extractor: bool=True,
                 freeze_binary_head: bool=False,
                 training_only_binary_head: bool=True
                ):
        super().__init__()
        self.num_classes = num_classes
        self.attention_size = attention_size
        print(f"feature extractor path is: {feature_extractor_path}")

        self.binary_head = BinaryHead(feature_extractor_path=feature_extractor_path, freeze_feature_extractor=freeze_feature_extractor, freeze_binary_head=freeze_binary_head)

        self.branch_body1 = BranchBody(feature_extractor_path=feature_extractor_path, attention_size=attention_size)
        self.branch_body2 = BranchBody(feature_extractor_path=feature_extractor_path, attention_size=attention_size)

        self.branch_body1_leaf1 = BranchLeaf(attention_size=attention_size, num_classes=num_classes[0, 0])
        self.branch_body1_leaf2 = BranchLeaf(attention_size=attention_size, num_classes=num_classes[0, 1])
        self.branch_body2_leaf1 = BranchLeaf(attention_size=attention_size, num_classes=num_classes[1, 0])
        self.branch_body2_leaf2 = BranchLeaf(attention_size=attention_size, num_classes=num_classes[1, 1])

        self.training_only_binary_head = training_only_binary_head

        if self.training_only_binary_head:
            self._freeze_all_for_binary_head_training()

    def forward(self, x):
        batch_size = x.shape[0]
        binary_output, embedding = self.binary_head(x)
        #if self.training_only_binary_head:
            #return binary_output, torch.zeros((batch_size, int(self.num_classes[0, 0]))), torch.zeros((batch_size, int(self.num_classes[0, 1]))), torch.zeros((batch_size, int(self.num_classes[1, 0]))), torch.zeros((batch_size, int(self.num_classes[1, 1])))
        
        x_1_1, x_1_2 = self.branch_body1(embedding)
        x_2_1, x_2_2 = self.branch_body2(embedding)

        y_1_1 = self.branch_body1_leaf1(x_1_1, x_1_2, x_2_1, x_2_2)
        y_1_2 = self.branch_body1_leaf2(x_1_2, x_1_1, x_2_1, x_2_2)
        y_2_1 = self.branch_body2_leaf1(x_2_1, x_2_2, x_1_1, x_1_2)
        y_2_2 = self.branch_body2_leaf2(x_2_2, x_1_2, x_1_1, x_1_2)

        return binary_output, y_1_1, y_1_2, y_2_1, y_2_2
    
    def _freeze_feature_extractor(self):
        self.binary_head._freeze_feature_extractor()

    def _freeze_binary_head(self):
        self.binary_head._freeze_binary_head()
    
    def _unfreeze_feature_extractor(self):
        self.binary_head._unfreeze_feature_extractor()

    def _unfreeze_binary_head(self):
        self.binary_head._unfreeze_binary_head()
    
    def _freeze_all_for_binary_head_training(self):
        # Freezing all parameters apart the binary head
        for param in self.branch_body1.parameters():
            param.requires_grad = False
        for param in self.branch_body2.parameters():
            param.requires_grad = False
        for param in self.branch_body1_leaf1.parameters():
            param.requires_grad = False
        for param in self.branch_body1_leaf2.parameters():
            param.requires_grad = False
        for param in self.branch_body2_leaf1.parameters():
            param.requires_grad = False
        for param in self.branch_body2_leaf2.parameters():
            param.requires_grad = False

    def _unfreeze_all_for_end_binary_head_training(self):
        # Freezing all parameters apart the binary head
        for param in self.branch_body1.parameters():
            param.requires_grad = True
        for param in self.branch_body2.parameters():
            param.requires_grad = True
        for param in self.branch_body1_leaf1.parameters():
            param.requires_grad = True
        for param in self.branch_body1_leaf2.parameters():
            param.requires_grad = True
        for param in self.branch_body2_leaf1.parameters():
            param.requires_grad = True
        for param in self.branch_body2_leaf2.parameters():
            param.requires_grad = True


class WeightedCrossEntropy(nn.Module):
    def __init__(self, loss_weights, num_classes):
        super(WeightedCrossEntropy, self).__init__()

        self.criterion1 = nn.CrossEntropyLoss()
        self.criterion2 = nn.CrossEntropyLoss()
        self.criterion3 = nn.CrossEntropyLoss()
        self.criterion4 = nn.CrossEntropyLoss()
        self.criterion5 = nn.CrossEntropyLoss()
        self.loss_weights = loss_weights
        self.num_classes = num_classes

    def forward(self, output, target): 
        loss = 0
        batch_size = target.shape[0]

        binary, y_1_1, y_1_2, y_2_1, y_2_2 = output
        
        t = target[:, 0] >= 0.5
        indices_positive = t.nonzero().squeeze(dim=1)
        
        t = target[:, 0] <= 0.5
        indices_negative = t.nonzero().squeeze(dim=1)
        
        len_negative = len(indices_negative)
        len_positive = len(indices_positive)
        
        loss += self.criterion1(binary, target[:, 0].long()) * self.loss_weights[0]
        
        if len_positive > 0:
            loss += self.criterion2(y_2_1[indices_positive,:], target[indices_positive, 3].long()) * self.loss_weights[1]
            loss += self.criterion3(y_2_2[indices_positive,:], target[indices_positive, 4].long()) * self.loss_weights[2]
        if len_negative > 0:
            loss += self.criterion4(y_1_1[indices_negative,:], target[indices_negative, 1].long()) * self.loss_weights[1]
            loss += self.criterion5(y_1_2[indices_negative,:], target[indices_negative, 2].long()) * self.loss_weights[2]
        
        return loss/batch_size

class SeggregatedBACNN(LightningModule, ABC):
    def __init__(   self,
                    results: Path,
                    model_name: str,
                    num_classes: list[int],
                    attention_size: int,
                    feature_extractor_path: Path,
                    freeze_feature_extractor: Optional[bool],
                    freeze_binary_head: Optional[bool],
                    training_only_binary_head: bool=True,
                    learning_rate: float = 1e-03,
                    scheduler: str = None,
                    scheduler_parameters: Optional[dict] = None,
                    loss_weights: list[float] = [0.33, 0.33, 0.34],
                ):
        super().__init__()


        # log hyperparameters
        self.save_hyperparameters() # ignore=[Model_path]
        self.RESULTS = results
        self.model_name = model_name
        self.num_classes = num_classes

        if training_only_binary_head:
            print('\nINFO: Training only Binary Head\n')
            self.criterion = WeightedCrossEntropy(loss_weights=[1., 0., 0.], num_classes=num_classes)
        else:
            print(f"\nINFO: Using loss_weights of {loss_weights}\n")
            self.criterion = WeightedCrossEntropy(loss_weights=loss_weights, num_classes=num_classes)

        # Learning rate
        self.learning_rate = learning_rate
        self.scheduler = scheduler
        self.scheduler_parameters = scheduler_parameters
        print(f"feature extracator path is: {feature_extractor_path}")
        # create model
        self.model = Modified_BA_CNN(feature_extractor_path=feature_extractor_path,num_classes=num_classes,attention_size=attention_size,freeze_feature_extractor=freeze_feature_extractor,freeze_binary_head=freeze_binary_head,training_only_binary_head=training_only_binary_head)

        # Define metrics
        binary_metrics = MetricCollection({
                                            'binary_head_acc': MulticlassAccuracy(num_classes=2, average='weighted'),
                                            'binary_head_f1': MulticlassF1Score(num_classes=2, average='macro'),
                                            'binary_head_pre': MulticlassPrecision(num_classes=2, average='macro'),
                                            'binary_head_rec': MulticlassRecall(num_classes=2, average='macro'),
        })
        metrics_1_1 = MetricCollection({
                                            'body_1_leaf_1_f1':MulticlassF1Score(num_classes=int(num_classes[0, 0]), average='macro'),
                                            'body_1_leaf_1_acc':MulticlassAccuracy(num_classes=int(num_classes[0, 0]), average='weighted'),
                                            'body_1_leaf_1_pre':MulticlassPrecision(num_classes=int(num_classes[0, 0]), average='macro'),
                                            'body_1_leaf_1_rec':MulticlassRecall(num_classes=int(num_classes[0, 0]), average='macro'),
        })
        metrics_1_2 = MetricCollection({
                                            'body_1_leaf_2_f1':MulticlassF1Score(num_classes=int(num_classes[0, 1]), average='macro'),
                                            'body_1_leaf_2_acc':MulticlassAccuracy(num_classes=int(num_classes[0, 1]), average='weighted'),
                                            'body_1_leaf_2_pre':MulticlassPrecision(num_classes=int(num_classes[0, 1]), average='macro'),
                                            'body_1_leaf_2_rec':MulticlassRecall(num_classes=int(num_classes[0, 1]), average='macro'),
        })
        metrics_2_1 = MetricCollection({
                                            'body_2_leaf_1_f1':MulticlassF1Score(num_classes=int(num_classes[1, 0]), average='macro'),
                                            'body_2_leaf_1_acc':MulticlassAccuracy(num_classes=int(num_classes[1, 0]), average='weighted'),
                                            'body_2_leaf_1_pre':MulticlassPrecision(num_classes=int(num_classes[1, 0]), average='macro'),
                                            'body_2_leaf_1_rec':MulticlassRecall(num_classes=int(num_classes[1, 0]), average='macro'),
        })
        metrics_2_2 = MetricCollection({
                                            'body_2_leaf_2_f1':MulticlassF1Score(num_classes=int(num_classes[1, 1]), average='macro'),
                                            'body_2_leaf_2_acc':MulticlassAccuracy(num_classes=int(num_classes[1, 1]), average='weighted'),
                                            'body_2_leaf_2_pre':MulticlassPrecision(num_classes=int(num_classes[1, 1]), average='macro'),
                                            'body_2_leaf_2_rec':MulticlassRecall(num_classes=int(num_classes[1, 1]), average='macro'),
                                        })
        
        self.train_binary_metrics = binary_metrics.clone(prefix='train_').to('cpu')
        self.train_metrics_1_1 = metrics_1_1.clone(prefix='train_').to('cpu')
        self.train_metrics_1_2 = metrics_1_2.clone(prefix='train_').to('cpu')
        self.train_metrics_2_1 = metrics_2_1.clone(prefix='train_').to('cpu')
        self.train_metrics_2_2 = metrics_2_2.clone(prefix='train_').to('cpu')
        """
        self.train_metrics_class = [train_binary_metrics,
                                    train_metrics_1_1,
                                    train_metrics_1_2,
                                    train_metrics_2_1,
                                    train_metrics_2_2,
                                   ]
        """
        
        self.val_binary_metrics = binary_metrics.clone(prefix='val_').to('cpu')
        self.val_metrics_1_1 = metrics_1_1.clone(prefix='val_').to('cpu')
        self.val_metrics_1_2 = metrics_1_2.clone(prefix='val_').to('cpu')
        self.val_metrics_2_1 = metrics_2_1.clone(prefix='val_').to('cpu')
        self.val_metrics_2_2 = metrics_2_2.clone(prefix='val_').to('cpu')
        """
        self.val_metrics_class = [val_binary_metrics,
                                   val_metrics_1_1,
                                   val_metrics_1_2,
                                   val_metrics_2_1,
                                   val_metrics_2_2,
                                   ]
        """
        self.test_binary_metrics = binary_metrics.clone(prefix='test_').to('cpu')
        self.test_metrics_1_1 = metrics_1_1.clone(prefix='test_').to('cpu')
        self.test_metrics_1_2 = metrics_1_2.clone(prefix='test_').to('cpu')
        self.test_metrics_2_1 = metrics_2_1.clone(prefix='test_').to('cpu')
        self.test_metrics_2_2 = metrics_2_2.clone(prefix='test_').to('cpu')
        
        """
        self.test_metrics_class = [test_binary_metrics,
                                    test_metrics_1_1,
                                    test_metrics_1_2,
                                    test_metrics_2_1,
                                    test_metrics_2_2,
                                   ]
        """
        
    def update_loss_weights(self, new_loss_weights):
        self.criterion = WeightedCrossEntropy(loss_weights=new_loss_weights, num_classes=self.num_classes)
    
    def freeze_feature_extractor(self):
        self.model._freeze_feature_extractor()

    def freeze_binary_head(self):
        self.model._freeze_binary_head()
    
    def freeze_all_for_binary_head_training(self):
        self.model._freeze_all_for_binary_head_training()
    
    def unfreeze_all_for_end_binary_head_training(self):
        self.model._unfreeze_all_for_end_binary_head_training()
        
    def _indices(self, targets):
        t = targets[:, 0] >= 0.5
        indices_positive = t.nonzero().squeeze(dim=1)
        
        t = targets[:, 0] <= 0.5
        indices_negative = t.nonzero().squeeze(dim=1)
        
        len_negative = len(indices_negative)
        len_positive = len(indices_positive)
        
        return indices_negative, indices_positive, len_negative, len_positive
        
       
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):

        # Get targets 
        images, class_name_target = batch
        device = images.device
        #class_name_target = torch.Tensor(class_name_target).to(device)
        #class_name_target = torch.Tensor(class_name_target)

        # Get predictions
        class_preds = self.forward(images)

        # Calculate loss
        loss = self.criterion(class_preds, class_name_target)

        
        self.log('train_loss', loss, sync_dist=True)
        
        return {"loss": loss,
               "c_true": class_name_target.to('cpu'),
               "c_pred_binary": class_preds[0].to('cpu'),
               "c_pred_1_1": class_preds[1].to('cpu'),
               "c_pred_1_2": class_preds[2].to('cpu'),
               "c_pred_2_1": class_preds[3].to('cpu'),
               "c_pred_2_2": class_preds[4].to('cpu')}


    def training_epoch_end(self, outputs):
        
        class_name_target = torch.cat([x["c_true"] for x in outputs], dim=0)
        class_preds_binary = torch.cat([x["c_pred_binary"] for x in outputs], dim=0)
        class_preds_1_1 = torch.cat([x["c_pred_1_1"] for x in outputs], dim=0)
        class_preds_1_2 = torch.cat([x["c_pred_1_2"] for x in outputs], dim=0)
        class_preds_2_1 = torch.cat([x["c_pred_2_1"] for x in outputs], dim=0)
        class_preds_2_2 = torch.cat([x["c_pred_2_2"] for x in outputs], dim=0)
        
        #print('train1', self.train_binary_metrics["binary_head_f1"].device)
        #print('train2', self.train_metrics_1_1["body_1_leaf_1_acc"].device)
        #print('train3', self.train_metrics_1_2["body_1_leaf_2_acc"].device)
        #print('train4', self.train_metrics_2_1["body_2_leaf_1_acc"].device)
        #print('train5', self.train_metrics_2_2["body_2_leaf_2_acc"].device)
        #print('class_name_target', class_name_target.device)
        #print('class_preds_binary', class_preds_binary.device)
        #print('class_preds_1_1', class_preds_1_1.device)
        #print('class_preds_1_2', class_preds_1_2.device)
        #print('class_preds_2_1', class_preds_2_1.device)
        #print('class_preds_2_2', class_preds_2_2.device)
        
        device = 'cpu'
        
        self.train_binary_metrics.to(device)
        self.train_metrics_1_1.to(device)
        self.train_metrics_1_2.to(device)
        self.train_metrics_2_1.to(device)
        self.train_metrics_2_2.to(device)

        
        self.train_binary_metrics(class_preds_binary, class_name_target[:, 0].long())
        indices_negative, indices_positive, len_negative, len_positive = self._indices(class_name_target)
        
        if len_negative != 0:
            self.train_metrics_1_1(torch.argmax(class_preds_1_1[indices_negative, :], dim=1), class_name_target[indices_negative, 1].long())
            self.train_metrics_1_2(torch.argmax(class_preds_1_2[indices_negative, :], dim=1), class_name_target[indices_negative, 2].long())
        if len_positive != 0:
            self.train_metrics_2_1(torch.argmax(class_preds_2_1[indices_positive, :], dim=1), class_name_target[indices_positive, 3].long())
            self.train_metrics_2_2(torch.argmax(class_preds_2_2[indices_positive, :], dim=1), class_name_target[indices_positive, 4].long())
        
        # calculate and log the metrics for classification
        train_binary_f1 = self.train_binary_metrics['binary_head_f1']
        self.log('train_binary_f1', train_binary_f1, on_step=False, on_epoch=True, sync_dist=True)
        train_binary_acc = self.train_binary_metrics['binary_head_acc']
        self.log('train_binary_acc', train_binary_acc, on_step=False, on_epoch=True, sync_dist=True)
        train_binary_rec = self.train_binary_metrics['binary_head_rec']
        self.log('train_binary_rec', train_binary_rec, on_step=False, on_epoch=True, sync_dist=True)
        train_binary_pre = self.train_binary_metrics['binary_head_pre']
        self.log('train_binary_pre', train_binary_pre, on_step=False, on_epoch=True, sync_dist=True)
        
        if len_negative != 0:
            train_1_1_f1 = self.train_metrics_1_1['body_1_leaf_1_f1']
            self.log('train_1_1_f1', train_1_1_f1, on_step=False, on_epoch=True, sync_dist=True)
            train_1_1_acc = self.train_metrics_1_1['body_1_leaf_1_acc']
            self.log('train_1_1_acc', train_1_1_acc, on_step=False, on_epoch=True, sync_dist=True)
            train_1_1_rec = self.train_metrics_1_1['body_1_leaf_1_rec']
            self.log('train_1_1_rec', train_1_1_rec, on_step=False, on_epoch=True, sync_dist=True)
            train_1_1_pre = self.train_metrics_1_1['body_1_leaf_1_pre']
            self.log('train_1_1_pre', train_1_1_pre, on_step=False, on_epoch=True, sync_dist=True)

            train_1_2_f1 = self.train_metrics_1_2['body_1_leaf_2_f1']
            self.log('train_1_2_f1', train_1_2_f1, on_step=False, on_epoch=True, sync_dist=True)
            train_1_2_acc = self.train_metrics_1_2['body_1_leaf_2_acc']
            self.log('train_1_2_acc', train_1_2_acc, on_step=False, on_epoch=True, sync_dist=True)
            train_1_2_rec = self.train_metrics_1_2['body_1_leaf_2_rec']
            self.log('train_1_2_rec', train_1_2_rec, on_step=False, on_epoch=True, sync_dist=True)
            train_1_2_pre = self.train_metrics_1_2['body_1_leaf_2_pre']
            self.log('train_1_2_pre', train_1_2_pre, on_step=False, on_epoch=True, sync_dist=True)
        
        if len_positive != 0:
            train_2_1_f1 = self.train_metrics_2_1['body_2_leaf_1_f1']
            self.log('train_2_1_f1', train_2_1_f1, on_step=False, on_epoch=True, sync_dist=True)
            train_2_1_acc = self.train_metrics_2_1['body_2_leaf_1_acc']
            self.log('train_2_1_acc', train_2_1_acc, on_step=False, on_epoch=True, sync_dist=True)
            train_2_1_rec = self.train_metrics_2_1['body_2_leaf_1_rec']
            self.log('train_2_1_rec', train_2_1_rec, on_step=False, on_epoch=True, sync_dist=True)
            train_2_1_pre = self.train_metrics_2_1['body_2_leaf_1_pre']
            self.log('train_2_1_pre', train_2_1_pre, on_step=False, on_epoch=True, sync_dist=True)

            train_2_2_f1 = self.train_metrics_2_2['body_2_leaf_2_f1']
            self.log('train_2_2_f1', train_2_2_f1, on_step=False, on_epoch=True, sync_dist=True)
            train_2_2_acc = self.train_metrics_2_2['body_2_leaf_2_acc']
            self.log('train_2_2_acc', train_2_2_acc, on_step=False, on_epoch=True, sync_dist=True)
            train_2_2_rec = self.train_metrics_2_2['body_2_leaf_2_rec']
            self.log('train_2_2_rec', train_2_2_rec, on_step=False, on_epoch=True, sync_dist=True)
            train_2_2_pre = self.train_metrics_2_2['body_2_leaf_2_pre']
            self.log('train_2_2_pre', train_2_2_pre, on_step=False, on_epoch=True, sync_dist=True)
        
        

    def validation_step(self, batch, batch_idx):

        # Get targets 
        images, class_name_target = batch
        device = images.device

        #class_name_target = torch.Tensor(class_name_target).to(device)

        # Get predictions
        class_preds = self.forward(images)

        # Calculate loss
        loss = self.criterion(class_preds, class_name_target)
        
  
        
        self.log('val_loss', loss, sync_dist=True)
        
        return {"val_loss": loss,
               "c_true": class_name_target.to('cpu'),
               "c_pred_binary": class_preds[0].to('cpu'),
               "c_pred_1_1": class_preds[1].to('cpu'),
               "c_pred_1_2": class_preds[2].to('cpu'),
               "c_pred_2_1": class_preds[3].to('cpu'),
               "c_pred_2_2": class_preds[4].to('cpu')}


    def validation_epoch_end(self, outputs):
        
        class_name_target = torch.cat([x["c_true"] for x in outputs], dim=0)
        class_preds_binary = torch.cat([x["c_pred_binary"] for x in outputs], dim=0)
        class_preds_1_1 = torch.cat([x["c_pred_1_1"] for x in outputs], dim=0)
        class_preds_1_2 = torch.cat([x["c_pred_1_2"] for x in outputs], dim=0)
        class_preds_2_1 = torch.cat([x["c_pred_2_1"] for x in outputs], dim=0)
        class_preds_2_2 = torch.cat([x["c_pred_2_2"] for x in outputs], dim=0)
        
        
        
        device = 'cpu'
        
        self.val_binary_metrics.to(device)
        self.val_metrics_1_1.to(device)
        self.val_metrics_1_2.to(device)
        self.val_metrics_2_1.to(device)
        
        self.val_metrics_2_2.to(device)
        
        
        self.val_binary_metrics(class_preds_binary, class_name_target[:, 0].long())
        indices_negative, indices_positive, len_negative, len_positive = self._indices(class_name_target)
        
        if len_negative != 0:
            self.val_metrics_1_1(torch.argmax(class_preds_1_1[indices_negative, :], dim=1), class_name_target[indices_negative, 1].long())
            self.val_metrics_1_2(torch.argmax(class_preds_1_2[indices_negative, :], dim=1), class_name_target[indices_negative, 2].long())
        if len_positive != 0:
            self.val_metrics_2_1(torch.argmax(class_preds_2_1[indices_positive, :], dim=1), class_name_target[indices_positive, 3].long())
            self.val_metrics_2_2(torch.argmax(class_preds_2_2[indices_positive, :], dim=1), class_name_target[indices_positive, 4].long())
        
        # calculate and log the metrics for classification
        val_binary_f1 = self.val_binary_metrics['binary_head_f1']
        self.log('val_binary_f1', val_binary_f1, on_step=False, on_epoch=True, sync_dist=True)
        val_binary_acc = self.val_binary_metrics['binary_head_acc']
        self.log('val_binary_acc', val_binary_acc, on_step=False, on_epoch=True, sync_dist=True)
        val_binary_rec = self.val_binary_metrics['binary_head_rec']
        self.log('val_binary_rec', val_binary_rec, on_step=False, on_epoch=True, sync_dist=True)
        val_binary_pre = self.val_binary_metrics['binary_head_pre']
        self.log('val_binary_pre', val_binary_pre, on_step=False, on_epoch=True, sync_dist=True)
        
        if len_negative != 0:
            val_1_1_f1 = self.val_metrics_1_1['body_1_leaf_1_f1']
            self.log('val_1_1_f1', val_1_1_f1, on_step=False, on_epoch=True, sync_dist=True)
            val_1_1_acc = self.val_metrics_1_1['body_1_leaf_1_acc']
            self.log('val_1_1_acc', val_1_1_acc, on_step=False, on_epoch=True, sync_dist=True)
            val_1_1_rec = self.val_metrics_1_1['body_1_leaf_1_rec']
            self.log('val_1_1_rec', val_1_1_rec, on_step=False, on_epoch=True, sync_dist=True)
            val_1_1_pre = self.val_metrics_1_1['body_1_leaf_1_pre']
            self.log('val_1_1_pre', val_1_1_pre, on_step=False, on_epoch=True, sync_dist=True)

            val_1_2_f1 = self.val_metrics_1_2['body_1_leaf_2_f1']
            self.log('val_1_2_f1', val_1_2_f1, on_step=False, on_epoch=True, sync_dist=True)
            val_1_2_acc = self.val_metrics_1_2['body_1_leaf_2_acc']
            self.log('val_1_2_acc', val_1_2_acc, on_step=False, on_epoch=True, sync_dist=True)
            val_1_2_rec = self.val_metrics_1_2['body_1_leaf_2_rec']
            self.log('val_1_2_rec', val_1_2_rec, on_step=False, on_epoch=True, sync_dist=True)
            val_1_2_pre = self.val_metrics_1_2['body_1_leaf_2_pre']
            self.log('val_1_2_pre', val_1_2_pre, on_step=False, on_epoch=True, sync_dist=True)
        
        if len_positive != 0:
            val_2_1_f1 = self.val_metrics_2_1['body_2_leaf_1_f1']
            self.log('val_2_1_f1', val_2_1_f1, on_step=False, on_epoch=True, sync_dist=True)
            val_2_1_acc = self.val_metrics_2_1['body_2_leaf_1_acc']
            self.log('val_2_1_acc', val_2_1_acc, on_step=False, on_epoch=True, sync_dist=True)
            val_2_1_rec = self.val_metrics_2_1['body_2_leaf_1_rec']
            self.log('val_2_1_rec', val_2_1_rec, on_step=False, on_epoch=True, sync_dist=True)
            val_2_1_pre = self.val_metrics_2_1['body_2_leaf_1_pre']
            self.log('val_2_1_pre', val_2_1_pre, on_step=False, on_epoch=True, sync_dist=True)

            val_2_2_f1 = self.val_metrics_2_2['body_2_leaf_2_f1']
            self.log('val_2_2_f1', val_2_2_f1, on_step=False, on_epoch=True, sync_dist=True)
            val_2_2_acc = self.val_metrics_2_2['body_2_leaf_2_acc']
            self.log('val_2_2_acc', val_2_2_acc, on_step=False, on_epoch=True, sync_dist=True)
            val_2_2_rec = self.val_metrics_2_2['body_2_leaf_2_rec']
            self.log('val_2_2_rec', val_2_2_rec, on_step=False, on_epoch=True, sync_dist=True)
            val_2_2_pre = self.val_metrics_2_2['body_2_leaf_2_pre']
            self.log('val_2_2_pre', val_2_2_pre, on_step=False, on_epoch=True, sync_dist=True)
        """
        if self.current_epoch == 1:
            c_true = torch.cat([x['c_target'] for x in outputs]).to('cpu')
            TRUE_FILE = Path(self.RESULTS) / 'c_true_first_epoch.pt' 
            embedding = torch.cat([x['embedding'] for x in outputs]).to('cpu')
            EMBEDDING_FILE = Path(self.RESULTS) / 'embedding_first_epoch.pt'

            torch.save(c_true, TRUE_FILE)
            torch.save(embedding, EMBEDDING_FILE)
        """
        
    def test_step(self, batch, batch_idx):
        # Get targets 
        images, class_name_target = batch
        device = images.device
        #class_name_target = torch.Tensor(class_name_target).to(device)

        # Get predictions
        class_preds = self.forward(images)

        # Calculate loss
        #loss = self.criterion(class_preds, class_name_target)
        

        #self.log('test_loss', loss, sync_dist=True)
        
        return {#"test_loss": loss,
               "c_true": class_name_target.to('cpu'),
               "c_pred_binary": class_preds[0].to('cpu'),
               "c_pred_1_1": class_preds[1].to('cpu'),
               "c_pred_1_2": class_preds[2].to('cpu'),
               "c_pred_2_1": class_preds[3].to('cpu'),
               "c_pred_2_2": class_preds[4].to('cpu')}
                

    def test_epoch_end(self, outputs):
        
        class_name_target = torch.cat([x["c_true"] for x in outputs if len(x["c_true"].size())==2], dim=0)
        class_preds_binary = torch.cat([x["c_pred_binary"] for x in outputs if len(x["c_pred_binary"].size())==2], dim=0)
        class_preds_1_1 = torch.cat([x["c_pred_1_1"] for x in outputs if len(x["c_pred_1_1"].size())==2], dim=0)
        class_preds_1_2 = torch.cat([x["c_pred_1_2"] for x in outputs if len(x["c_pred_1_2"].size())==2], dim=0)
        class_preds_2_1 = torch.cat([x["c_pred_2_1"] for x in outputs if len(x["c_pred_2_1"].size())==2], dim=0)
        class_preds_2_2 = torch.cat([x["c_pred_2_2"] for x in outputs if len(x["c_pred_2_2"].size())==2], dim=0)
        
        print(f'length is {class_name_target.size()}')
        print(f'length is {class_preds_binary.size()}')
        print(f'length is {class_preds_1_1.size()}')
        print(f'length is {class_preds_1_2.size()}')
        print(f'length is {class_preds_2_1.size()}')
        print(f'length is {class_preds_2_2.size()}')
        
        TRUE_FILE = Path(self.RESULTS) / f'c_true_test.pt' 
        PRED_BINARY = Path(self.RESULTS) / f'c_pred_binary.pt'
        PRED_1_1 = Path(self.RESULTS) / f'c_pred_1_1.pt'
        PRED_1_2 = Path(self.RESULTS) / f'c_pred_1_2.pt'
        PRED_2_1 = Path(self.RESULTS) / f'c_pred_2_1.pt'
        PRED_2_2 = Path(self.RESULTS) / f'c_pred_2_2.pt'

        torch.save(class_name_target, TRUE_FILE)
        torch.save(class_preds_binary, PRED_BINARY)
        torch.save(class_preds_1_1, PRED_1_1)
        torch.save(class_preds_1_2, PRED_1_2)
        torch.save(class_preds_2_1, PRED_2_1)
        torch.save(class_preds_2_2, PRED_2_2)
        
        
        device = 'cpu'
        
        self.test_binary_metrics.to(device)
        self.test_metrics_1_1.to(device)
        self.test_metrics_1_2.to(device)
        self.test_metrics_2_1.to(device)
        self.test_metrics_2_2.to(device)
        
        self.test_binary_metrics(class_preds_binary, class_name_target[:, 0].long())
        indices_negative, indices_positive, len_negative, len_positive = self._indices(class_name_target)
        
        if len_negative != 0:
            self.test_metrics_1_1(torch.argmax(class_preds_1_1[indices_negative,:], dim=1), class_name_target[indices_negative, 1].long())
            self.test_metrics_1_2(torch.argmax(class_preds_1_2[indices_negative,:], dim=1), class_name_target[indices_negative, 2].long())
        if len_positive != 0:
            self.test_metrics_2_1(torch.argmax(class_preds_2_1[indices_positive,:], dim=1), class_name_target[indices_positive, 3].long())
            self.test_metrics_2_2(torch.argmax(class_preds_2_2[indices_positive,:], dim=1), class_name_target[indices_positive, 4].long())
        
        # calculate and log the metrics for classification
        test_binary_f1 = self.test_binary_metrics['binary_head_f1']
        self.log('test_binary_f1', test_binary_f1, on_step=False, on_epoch=True, sync_dist=True)
        test_binary_acc = self.test_binary_metrics['binary_head_acc']
        self.log('test_binary_acc', test_binary_acc, on_step=False, on_epoch=True, sync_dist=True)
        test_binary_rec = self.test_binary_metrics['binary_head_rec']
        self.log('test_binary_rec', test_binary_rec, on_step=False, on_epoch=True, sync_dist=True)
        test_binary_pre = self.test_binary_metrics['binary_head_pre']
        self.log('test_binary_pre', test_binary_pre, on_step=False, on_epoch=True, sync_dist=True)
        
        if len_negative != 0:
            test_1_1_f1 = self.test_metrics_1_1['body_1_leaf_1_f1']
            self.log('test_1_1_f1', test_1_1_f1, on_step=False, on_epoch=True, sync_dist=True)
            test_1_1_acc = self.test_metrics_1_1['body_1_leaf_1_acc']
            self.log('test_1_1_acc', test_1_1_acc, on_step=False, on_epoch=True, sync_dist=True)
            test_1_1_rec = self.test_metrics_1_1['body_1_leaf_1_rec']
            self.log('test_1_1_rec', test_1_1_rec, on_step=False, on_epoch=True, sync_dist=True)
            test_1_1_pre = self.test_metrics_1_1['body_1_leaf_1_pre']
            self.log('test_1_1_pre', test_1_1_pre, on_step=False, on_epoch=True, sync_dist=True)

            test_1_2_f1 = self.test_metrics_1_2['body_1_leaf_2_f1']
            self.log('test_1_2_f1', test_1_2_f1, on_step=False, on_epoch=True, sync_dist=True)
            test_1_2_acc = self.test_metrics_1_2['body_1_leaf_2_acc']
            self.log('test_1_2_acc', test_1_2_acc, on_step=False, on_epoch=True, sync_dist=True)
            test_1_2_rec = self.test_metrics_1_2['body_1_leaf_2_rec']
            self.log('test_1_2_rec', test_1_2_rec, on_step=False, on_epoch=True, sync_dist=True)
            test_1_2_pre = self.test_metrics_1_2['body_1_leaf_2_pre']
            self.log('test_1_2_pre', test_1_2_pre, on_step=False, on_epoch=True, sync_dist=True)
        
        if len_positive != 0:
            test_2_1_f1 = self.test_metrics_2_1['body_2_leaf_1_f1']
            self.log('test_2_1_f1', test_2_1_f1, on_step=False, on_epoch=True, sync_dist=True)
            test_2_1_acc = self.test_metrics_2_1['body_2_leaf_1_acc']
            self.log('test_2_1_acc', test_2_1_acc, on_step=False, on_epoch=True, sync_dist=True)
            test_2_1_rec = self.test_metrics_2_1['body_2_leaf_1_rec']
            self.log('test_2_1_rec', test_2_1_rec, on_step=False, on_epoch=True, sync_dist=True)
            test_2_1_pre = self.test_metrics_2_1['body_2_leaf_1_pre']
            self.log('test_2_1_pre', test_2_1_pre, on_step=False, on_epoch=True, sync_dist=True)

            test_2_2_f1 = self.test_metrics_2_2['body_2_leaf_2_f1']
            self.log('test_2_2_f1', test_2_2_f1, on_step=False, on_epoch=True, sync_dist=True)
            test_2_2_acc = self.test_metrics_2_2['body_2_leaf_2_acc']
            self.log('test_2_2_acc', test_2_2_acc, on_step=False, on_epoch=True, sync_dist=True)
            test_2_2_rec = self.test_metrics_2_2['body_2_leaf_2_rec']
            self.log('test_2_2_rec', test_2_2_rec, on_step=False, on_epoch=True, sync_dist=True)
            test_2_2_pre = self.test_metrics_2_2['body_2_leaf_2_pre']
            self.log('test_2_2_pre', test_2_2_pre, on_step=False, on_epoch=True, sync_dist=True)
        

    def configure_optimizers(self):
        print('#### Configuring Optimizer ####')
        optimizer = torch.optim.AdamW(params=self.parameters(), lr = self.learning_rate)
        print(f'Using a learning rate of {self.learning_rate}')
        
        if self.scheduler == 'cosine_annealing':
            print('#### Using Cosine Annealing Scheduler ####')
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                T_max=self.scheduler_parameters['T_max'],
                                                eta_min=self.scheduler_parameters['eta_min'],
                                                                  verbose = True)
            return {"optimizer": optimizer, "lr_scheduler": scheduler}
        
        elif self.scheduler == 'reduce_lr_on_plateau':
            print('#### Using Reduce On Plateau Scheduler ####')
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                                   'max',
                                                                   factor=0.5,
                                                            patience=self.scheduler_parameters['patience'],
                                                                  min_lr=self.scheduler_parameters['min_lr'],
                                                                  verbose = True)
            return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_binary_f1"}
        
        elif self.scheduler == 'exponential_lr':
            print('#### Using Exponential Scheduler ####')
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer,
                                                                  gamma=self.scheduler_parameters['gamma'],
                                                                  verbose = True)
            return {"optimizer": optimizer, "lr_scheduler": scheduler}
        
        else:
            return optimizer
        
    def update_learning_rate(self, new_learning_rate):
        self.learning_rate = new_learning_rate
        for param_group in self.optimizers:
            param_group['lr'] = new_learning_rate