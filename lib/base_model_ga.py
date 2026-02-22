import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from sklearn.metrics import classification_report, precision_recall_fscore_support, accuracy_score
import wandb
from tqdm import tqdm
import torch.nn.functional as F
import pandas as pd


class Model:
    """Base model class for training and evaluation"""

    def __init__(self):
        self.params = None
        self.preprocessor = None
        self.dataset_class = None
        self.collate_fn = None

    def evaluate_single_batch(self, batch, model, device):
        raise NotImplementedError("This is implemented for subclasses only")

    def create_dataset(self, X_preprocessed, y):
        dataset = self.dataset_class(X_preprocessed, y)
        return dataset

    def create_datasetAB(self, A_preprocessed, B_preprocessed, y):
        dataset = self.dataset_class(A_preprocessed, B_preprocessed, y)
        return dataset

    def fit(self, X_train, y_train, epochs=3):
        number_of_train_samples = len(X_train)
        # Text preprocessing
        X_train_preprocessed = self.preprocessor.preprocess(X_train)
        # Creating train dataset
        train_dataset = self.create_dataset(X_train_preprocessed, y_train)
        # Creating train dataloader
        train_dataloader = create_train_dataloader(
            train_dataset, self.params['batch_size'], self.collate_fn)
        # Training
        self.train_for_epochs(
            number_of_train_samples,
            train_dataloader,
            epochs)

    def fit(self, A_train, B_train, y_train, epochs=3):
        number_of_train_samples = len(A_train)
        # Text preprocessing
        X_train_preprocessed = self.preprocessor.preprocess(A_train, B_train)
        # Creating train dataset
        train_dataset = self.create_dataset(X_train_preprocessed, y_train)
        # Creating train dataloader
        train_dataloader = create_train_dataloader(
            train_dataset, self.params['batch_size'], self.collate_fn)
        # Training
        self.train_for_epochs(
            number_of_train_samples,
            train_dataloader,
            epochs)

    def train_and_evaluate(self, X_train, X_test, y_train, y_test, args):
        number_of_train_samples = len(X_train)
        number_of_test_samples = len(X_test)
        X_train_preprocessed = self.preprocessor.preprocess(X_train)
        X_test_preprocessed = self.preprocessor.preprocess(X_test)
        train_dataset = self.create_dataset(X_train_preprocessed, y_train)
        test_dataset = self.create_dataset(X_test_preprocessed, y_test)
        train_dataloader = create_train_dataloader(
            train_dataset, self.params['batch_size'], self.collate_fn)
        test_dataloader = create_test_dataloader(
            test_dataset, self.params['batch_size'], self.collate_fn)
        result = self.train_and_evaluate_preprocessed(
            number_of_train_samples,
            train_dataloader,
            number_of_test_samples,
            test_dataloader,
            len(test_dataloader),
            args)
        return result

    def train_and_evaluate(self, A_train, B_train, A_eval, B_eval, y_train, y_eval, args):
        number_of_train_samples = len(A_train)
        number_of_test_samples = len(A_eval)
        X_train_preprocessed = self.preprocessor.preprocess(A_train, B_train)
        X_test_preprocessed = self.preprocessor.preprocess(A_eval, B_eval)
        train_dataset = self.create_dataset(X_train_preprocessed, y_train)
        test_dataset = self.create_dataset(X_test_preprocessed, y_eval)
        train_dataloader = create_train_dataloader(
            train_dataset, self.params['batch_size'], self.collate_fn)
        test_dataloader = create_test_dataloader(
            test_dataset, self.params['batch_size'], self.collate_fn)
        result = self.train_and_evaluate_preprocessed(
            number_of_train_samples,
            train_dataloader,
            number_of_test_samples,
            test_dataloader,
            len(test_dataloader),
            args)
        return result
    
    def train_and_evaluateAplusB(self, paras_tr, comms_tr, paras_dev, comms_dev, labels_tr, labels_dev, drafts_dev, wgs_dev, args):
        number_of_train_samples = len(labels_tr)
        number_of_test_samples = len(labels_dev)

        X_train_preprocessed = self.preprocessor.preprocess(paras_tr, comms_tr)
        X_test_preprocessed = self.preprocessor.preprocess(paras_dev, comms_dev)

        train_dataset = self.create_dataset(X_train_preprocessed, labels_tr)
        test_dataset = self.create_dataset(X_test_preprocessed, labels_dev)

        train_dataloader = create_train_dataloader(
            train_dataset, self.params['batch_size'], self.collate_fn)
        test_dataloader = create_test_dataloader(
            test_dataset, self.params['batch_size'], self.collate_fn)
            
        num_steps = args.epochs * len(train_dataloader)
        warm_steps = num_steps // args.epochs
        warm = torch.optim.lr_scheduler.LinearLR(self.optimizer, start_factor=1e-80, end_factor=1, total_iters=warm_steps)
        decayer = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, num_steps - warm_steps)
        scheduler = torch.optim.lr_scheduler.SequentialLR(self.optimizer, [warm, decayer], [warm_steps])
        self.scheduler = scheduler

        result = self.train_and_evaluate_preprocessed(
            number_of_train_samples,
            train_dataloader,
            number_of_test_samples,
            test_dataloader,
            len(test_dataloader),
            drafts_dev,
            wgs_dev,
            args)
        return result
    
    def train_and_evaluateAB(self, paras_tr, comms_tr, paras_dev, comms_dev, labels_tr, labels_dev, drafts_dev, wgs_dev, args):
        number_of_train_samples = len(labels_tr)
        number_of_test_samples = len(labels_dev)

        A_train_preprocessed = self.preprocessor.preprocess(paras_tr)
        B_train_preprocessed = self.preprocessor.preprocess(comms_tr)

        train_dataset = self.create_datasetAB(A_train_preprocessed, B_train_preprocessed, labels_tr)

        train_dataloader = create_train_dataloader(
            train_dataset, self.params['batch_size'], self.collate_fn)

        A_test_preprocessed = self.preprocessor.preprocess(paras_dev)
        B_test_preprocessed = self.preprocessor.preprocess(comms_dev)

        test_dataset = self.create_datasetAB(A_test_preprocessed, B_test_preprocessed, labels_dev)
        test_dataloader = create_test_dataloader(
            test_dataset, self.params['batch_size'], self.collate_fn)

        num_steps = args.epochs * len(train_dataloader)
    
        warm_steps = num_steps // args.epochs
        warm = torch.optim.lr_scheduler.LinearLR(self.optimizer, start_factor=1e-80, end_factor=1, total_iters=warm_steps)
        decayer = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, num_steps - warm_steps)
        scheduler = torch.optim.lr_scheduler.SequentialLR(self.optimizer, [warm, decayer], [warm_steps])
        self.scheduler = scheduler

        result = self.train_and_evaluate_preprocessed(
            number_of_train_samples,
            train_dataloader,
            number_of_test_samples,
            test_dataloader,
            len(train_dataloader),
            drafts_dev,
            wgs_dev,
            args)
                        
        return result

    def train_and_evaluate_preprocessed(
            self,
            number_of_train_samples,
            train_dataloader,
            number_of_test_samples,
            test_dataloader,
            num_steps,
            drafts_dev=None,
            wgs_dev=None,
            args=None):

        result = {
            'eval_acc': [],
            'eval_loss': [],
            'eval_f1': [],
            'eval_precision': [],
            'eval_recall': []}

        # best_model = SaveBestModel()
        early_stopping = EarlyStopping(patience=args.patience, verbose=args.verbose)

        for epoch in tqdm(list(range(args.epochs))):
            self.train_single_epoch(number_of_train_samples, train_dataloader, args)
            eval_loss, accuracy, metrics, _, _  = self.evaluate_single_epoch(
                number_of_test_samples, test_dataloader, epoch, drafts_dev, wgs_dev, args)
            eval_f1, eval_precision, eval_recall = metrics['f1'], metrics['precision'], metrics['recall']

            print(
                f'Epoch: {epoch}, \n'
                f'Eval accuracy: {accuracy:.6f}, \n'
                f'Eval loss: {eval_loss:.6f}, \n'
                f'Eval f1: {eval_f1:.6f}, \n'
                f'Eval precision: {eval_precision:.6f}, \n' 
                f'Eval recall: {eval_recall:.6f}')
            
            result['eval_acc'].append(accuracy)
            result['eval_loss'].append(eval_loss)
            result['eval_f1'].append(eval_f1)
            result['eval_precision'].append(eval_precision)
            result['eval_recall'].append(eval_recall)

            early_stopping(eval_loss, eval_f1, epoch, self.nn, args)
            # print("steps", num_steps*(epoch+1))
            wandb.log(data={
                "valid/loss": eval_loss,
                "valid/f1": eval_f1,
                "valid/precision": eval_precision,
                "valid/recall": eval_recall
            },  step=wandb.run.step, commit=True)
            # step=num_steps*(epoch+1), commit=True)

        return result

    def predict(self, X_test, y_test=None):
        test_samples = len(X_test)
        if y_test is None:
            y_test = [0] * len(X_test)  # dummy labels
        X_test_preprocessed = self.preprocessor.preprocess(X_test)
        test_dataset = self.create_dataset(X_test_preprocessed, y_test)
        test_dataloader = create_test_dataloader(
            test_dataset, self.params['batch_size'], self.collate_fn)
        _, _, metrics, _, _ = self.evaluate_single_epoch(
            test_samples, test_dataloader)
        return metrics

    def predict(self, A_test, B_test, y_test, args):
        test_samples = len(A_test)
        if y_test is None:
            y_test = [0] * len(A_test)  # dummy labels
        X_test_preprocessed = self.preprocessor.preprocess(A_test, B_test)
        test_dataset = self.create_dataset(X_test_preprocessed, y_test)
        test_dataloader = create_test_dataloader(
            test_dataset, self.params['batch_size'], self.collate_fn)

        _, _, metrics, _, _ = self.evaluate_single_epoch(
            test_samples, test_dataloader, args)
        return metrics

    def predictAB(self, A_test, B_test, y_test, args):
        test_samples = len(A_test)
        if y_test is None:
            y_test = [0] * len(A_test)  # dummy labels
        A_test_preprocessed = self.preprocessor.preprocess(A_test)
        B_test_preprocessed = self.preprocessor.preprocess(B_test)
        
        test_dataset = self.create_datasetAB(A_test_preprocessed, B_test_preprocessed, y_test)

        test_dataloader = create_test_dataloader(
            test_dataset, self.params['batch_size'], self.collate_fn)

        _, _, metrics, preds_total, _ = self.evaluate_single_epoch(
            test_samples, test_dataloader, args)
        
        return metrics, preds_total    
    
    def train_single_epoch(self, number_of_train_samples, train_dataloader, args=None):
        model = self.nn
        gradient_accumulation_steps = 2
        criterion = nn.BCEWithLogitsLoss()

        model.train()
        for i, batch in enumerate(train_dataloader):
            preds, labels = self.evaluate_single_batch(batch, model, self.params['device'])

            if args and (args.model_type == 'AB' or args.model_type == 'SBERT'):
                preds = torch.squeeze(preds, dim=1)

            loss = criterion(preds, labels.float())
            loss = loss / gradient_accumulation_steps
            loss.backward()

            if (i + 1) % gradient_accumulation_steps == 0:
                wandb.log({'train/step_loss': loss.item()}, step=wandb.run.step, commit=True)
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()

            del preds, labels
            torch.cuda.empty_cache()

    def train_for_epochs(
            self,
            number_of_train_samples,
            train_dataloader,
            epochs=3):
        for epoch in range(epochs):
            avg_loss, accuracy, metrics = self.train_single_epoch(
                number_of_train_samples, train_dataloader)
            f1, precision, recall = metrics['f1'], metrics['precision'], metrics['recall']
            print(
                f'Epoch: {epoch}, Train accuracy: {accuracy}, Train loss: {avg_loss}, Train f1: {f1}, Train precision: {precision}, Train recall: {recall}')

    def evaluate_single_epoch(self, test_samples, test_dataloader, epoch, drafts_dev=None, wgs_dev=None, args=None):
        model = self.nn
        total_loss = 0
        total_accurate = 0

        preds_total = []
        labels_total = []

        criterion = nn.BCEWithLogitsLoss() 
   
        model.eval()
        for _, batch in enumerate(test_dataloader):
            with torch.no_grad():
    
                preds, labels = self.evaluate_single_batch(
                    batch, model, self.params['device'])
                
                preds_total.extend(preds.detach())
                labels_total.extend(labels)
      
                if args and (args.model_type == 'AB' or args.model_type == 'SBERT'):
                    preds = torch.squeeze(preds, dim=1)

                loss = criterion(preds, labels)
                total_accurate, total_loss = calc_loss_and_accuracy(
                    loss, preds, labels, total_loss, total_accurate)
                
                # wandb.log({"val/step_loss": loss})
                del preds, labels
                torch.cuda.empty_cache()

        metrics = binary_label_metric(preds_total, labels_total, epoch, drafts_dev, wgs_dev, args)

        # compute the evaluation loss of the epoch
        preds_total = [x.item() for x in preds_total]
        labels_total = [x.item() for x in labels_total]
        avg_loss = total_loss / test_samples
        accuracy = total_accurate / test_samples

        return avg_loss, accuracy, metrics, preds_total, labels_total

class SaveBestModel:
    """Save the best model during training based on validation metrics"""
    def __init__(self, best_valid_loss=float('inf'), best_valid_f1=-0.1):
        self.best_valid_loss = best_valid_loss
        self.best_valid_f1 = best_valid_f1
        self.best_epoch = None
        
    def __call__(self, current_valid_loss, current_valid_f1, epoch, model, args):
        if (current_valid_f1 > self.best_valid_f1) or \
           (current_valid_f1 == self.best_valid_f1 and current_valid_loss < self.best_valid_loss):
            self.best_valid_loss = current_valid_loss
            self.best_valid_f1 = current_valid_f1
            self.best_epoch = epoch
            print(f"\nBest validation loss: {self.best_valid_loss}")
            print(f"\nBest validation f1: {self.best_valid_f1}")
            print(f"\nSaving best model for epoch: {self.best_epoch}\n")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
            }, args.model_path)
        else:
            print("This epoch has no better performance!")

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, epoch=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved. Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement. Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.early_stop = False
        self.delta = delta
        self.best_valid_loss = np.inf
        self.best_valid_f1 = -0.1
        self.best_epoch = epoch

    def __call__(self, current_valid_loss, current_valid_f1, epoch, model, args):
        if self.best_valid_f1 is None:
            self.best_valid_f1 = current_valid_f1
            self.save_checkpoint(epoch, model, args)
        elif current_valid_f1 < self.best_valid_f1 + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_valid_loss = current_valid_loss
            self.best_valid_f1 = current_valid_f1
            self.best_epoch = epoch
            self.save_checkpoint(epoch, model, args)
            self.counter = 0
    
    def save_checkpoint(self, epoch, model, args):
        if self.verbose:
            print(f"\nBest validation loss: {self.best_valid_loss:.6f}")
            print(f"\nBest validation f1: {self.best_valid_f1:.6f}")
            print(f"\nSaving best model for epoch: {self.best_epoch}\n")

        torch.save({'model_state_dict': model.state_dict()}, args.model_path)

def create_dataloader_batch(data, sampler_class, batch_size, collate_fn=None):
    """Create a DataLoader with given sampler and configuration"""
    sampler = sampler_class(data)
    dataloader = DataLoader(data, sampler=sampler, batch_size=batch_size, collate_fn=collate_fn)
    return dataloader


def create_train_dataloader(train_data, batch_size, collate_fn=None):
    """Create training DataLoader"""
    return create_dataloader_batch(train_data, RandomSampler, batch_size, collate_fn)


def create_test_dataloader(test_data, batch_size, collate_fn=None):
    """Create test/validation DataLoader"""
    return create_dataloader_batch(test_data, SequentialSampler, batch_size, collate_fn)


def create_dataloaders(train_data, test_data, batch_size, collate_fn=None):
    """Create both train and test DataLoaders"""
    train_dataloader = create_train_dataloader(train_data, batch_size, collate_fn)
    test_dataloader = create_test_dataloader(test_data, batch_size, collate_fn)
    return train_dataloader, test_dataloader


def calc_loss_and_accuracy(loss, preds, labels, total_loss, total_accurate, penalty=False):
    """Calculate accumulated loss and accuracy metrics"""
    total_loss = total_loss + loss.detach().cpu().numpy()
    predicted_classes = (preds.detach().numpy() >= 0.5)
    accurate = sum(predicted_classes == np.array(labels).astype(bool))
    total_accurate = total_accurate + accurate
    return total_accurate, total_loss

def binary_label_metric(predictions, references, epoch=None, drafts=None, wgs=None, args=None):
    """Compute binary classification metrics"""
    y_true = references
    m = torch.nn.Sigmoid()
    threshold = 0.5
    y_pred = [1 if m(p).numpy() >= threshold else 0 for p in predictions]
    
    dict_report = classification_report(y_true, y_pred, zero_division=1)
    print(dict_report)
    
    precision, recall, f1score, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=1)
    accuracy = accuracy_score(y_true, y_pred)

    preds_total = [m(x).item() for x in predictions]
    labels_total = [x.item() for x in references]

    if drafts is not None and wgs is not None:
        df = pd.DataFrame({"pred": preds_total, "label": labels_total, "draft": drafts, "wg": wgs})
        df.to_csv(f"{args.eval_path}/group.{epoch}.csv", sep='\t')

    metrics = {
        "f1": f1score,
        "precision": precision,
        "recall": recall
    }

    return metrics