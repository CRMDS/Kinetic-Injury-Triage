#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 23:54:06 2025
@model: Bio_ClinicalBERTClassifier
author: Midhun Shyam (M.Shyam)
"""

import os
import time
import pandas as pd
import torch
from torch.optim import AdamW, Adam, SGD
from torch.cuda.amp import autocast, GradScaler
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import numpy as np


class BioClinicalBERTClassifier:
    def __init__(
        self,
        model_name="emilyalsentzer/Bio_ClinicalBERT",
        num_labels=2,
        optimizer_class=AdamW,
        optimizer_params={'lr': 1e-5, 'weight_decay': 0.1},
        verbose=True,
        seed=8,
        batch_size=16,
        dropout_prob=None
    ):
        self.model_name = model_name
        self.num_labels = num_labels
        self.verbose = verbose
        self.seed = seed
        self.batch_size = batch_size
        self.dropout_prob = dropout_prob

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, clean_up_tokenization_spaces=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

        if dropout_prob is not None:
            self.model.config.hidden_dropout_prob = dropout_prob
            self.model.config.attention_probs_dropout_prob = dropout_prob

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.optimizer_class = optimizer_class
        self.optimizer_params = optimizer_params
        self.configure_optimizer()

        self.freeze_model_layers()
        self.unfreeze_classifier_layer()
        self.num_unfrozen_bert_layers = self.count_unfrozen_bert_layers()

        self._initial_state_dict = self.model.state_dict().copy()

    def configure_optimizer(self):
        self.optimizer = self.optimizer_class(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            **self.optimizer_params
        )

    def freeze_model_layers(self, requires_grad=False):
        for param in self.model.parameters():
            param.requires_grad = requires_grad

    def unfreeze_classifier_layer(self, requires_grad=True):
        self.model.classifier.weight.requires_grad = requires_grad
        self.model.classifier.bias.requires_grad = requires_grad

    def unfreeze_last_layers(self, n=1):
        total = len(self.model.bert.encoder.layer)
        if n > total:
            raise ValueError(f"Cannot unfreeze {n} layers; max is {total}.")
        for i in range(total - n, total):
            for p in self.model.bert.encoder.layer[i].parameters():
                p.requires_grad = True
        self.num_unfrozen_bert_layers = self.count_unfrozen_bert_layers()

    def count_unfrozen_bert_layers(self):
        return sum(
            1 for layer in self.model.bert.encoder.layer
            if any(p.requires_grad for p in layer.parameters())
        )

    def check_layer_status(self):
        for name, param in self.model.named_parameters():
            status = 'True' if param.requires_grad else 'False'
            print(f"{name}: requires_grad={status}")

    def dataframe_to_dataloader(self, df, shuffle=True,
                                text_column="TEXT", label_column="LABEL",
                                max_length=512):
        if text_column not in df or label_column not in df:
            raise ValueError(f"DataFrame needs '{text_column}' and '{label_column}' columns")
        df = df.copy()
        df[label_column] = df[label_column].astype(int)
        enc = self.tokenizer(
            df[text_column].tolist(),
            truncation=True, padding=True,
            max_length=max_length, return_tensors="pt"
        )
        labels = torch.tensor(df[label_column].tolist(), dtype=torch.long)
        ds = TensorDataset(enc['input_ids'], enc['attention_mask'], labels)
        return DataLoader(ds, batch_size=self.batch_size, shuffle=shuffle)

    def _run_train_epoch(
        self, dataset, num_epochs=3, test_split=0.2,
        early_stop_patience=3, shuffle_train=True,
        text_column="TEXT", label_column="LABEL",
        debug=True, print_every=1, primary_key=None
    ):
        scaler = GradScaler()

        cfg = {
            'layers_unlocked': self.num_unfrozen_bert_layers,
            'optimiser': self.optimizer_class.__name__,
            'seed': self.seed,
            'lr': self.optimizer_params.get('lr'),
            'weight_decay': self.optimizer_params.get('weight_decay'),
            'dropout': self.dropout_prob,
            'batch_size': self.batch_size,
        }
        cfg_str = '_'.join(str(v) for v in cfg.values())

        train_df, val_df = train_test_split(
            dataset, test_size=test_split, random_state=self.seed,
            stratify=dataset[label_column]
        )
        if debug:
            print(f"Train: {train_df.shape}, Val: {val_df.shape}")

        train_loader = self.dataframe_to_dataloader(
            train_df, shuffle=shuffle_train,
            text_column=text_column, label_column=label_column
        )
        val_loader = self.dataframe_to_dataloader(
            val_df, shuffle=False,
            text_column=text_column, label_column=label_column
        )

        epochs, train_losses, val_losses, train_accs, val_accs = [], [], [], [], []
        total_train_time = total_val_time = 0.0
        best_val = float('inf')
        best_weights = None
        no_improve = 0
        early_stop_triggered = False

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.1,
            patience=3, threshold=0.001, verbose=debug
        )
        final_metrics = {}

        for epoch in range(1, num_epochs + 1):
            epochs.append(epoch)
            self.model.train()
            t0 = time.time()
            loss_sum = 0.0
            for ids, masks, labs in train_loader:
                ids, masks, labs = ids.to(self.device), masks.to(self.device), labs.to(self.device)
                self.optimizer.zero_grad()
                with autocast():
                    out = self.model(ids, attention_mask=masks, labels=labs)
                loss = out.loss
                scaler.scale(loss).backward()
                scaler.step(self.optimizer)
                scaler.update()
                loss_sum += loss.item()
            t1 = time.time()
            total_train_time += t1 - t0
            avg_train = loss_sum / len(train_loader)
            train_losses.append(avg_train)
            _, tr_metric = self.evaluate_loss(train_loader, use_amp=False)
            train_accs.append(tr_metric['accuracy'])

            v0 = time.time()
            val_loss, val_metric = self.evaluate_loss(val_loader, use_amp=True)
            v1 = time.time()
            total_val_time += v1 - v0
            val_losses.append(val_loss)
            val_accs.append(val_metric['accuracy'])
            final_metrics = val_metric
            scheduler.step(val_loss)

            if val_loss < best_val:
                best_val = val_loss
                best_weights = self.model.state_dict().copy()
                no_improve = 0
            else:
                no_improve += 1

            if debug and epoch % print_every == 0:
                print(f"Epoch {epoch} | train_loss={avg_train:.4f}, train_acc={train_accs[-1]:.4f} | "
                      f"val_loss={val_losses[-1]:.4f}, val_acc={val_accs[-1]:.4f}")

            if no_improve >= early_stop_patience:
                early_stop_triggered = True
                if debug:
                    print(f"Early stopping at epoch {epoch}")
                break

        if best_weights is not None:
            self.model.load_state_dict(best_weights)
            if debug:
                print(f"Best val loss: {best_val:.4f}")

        metrics_df = pd.DataFrame({
            'epoch': epochs,
            'train_loss': train_losses,
            'train_accuracy': train_accs,
            'val_loss': val_losses,
            'val_accuracy': val_accs
        })
        metrics_file = f"{cfg_str}.csv"
        metrics_df.to_csv(metrics_file, index=False)
        print(f"Saved per-epoch metrics to {metrics_file}")

        cm = confusion_matrix(final_metrics['true_labels'], final_metrics['predictions'])
        tn, fp, fn, tp = cm.ravel()

        summary = {
            **cfg,
            'epochs_ran': len(epochs),
            'early_stop_triggered': early_stop_triggered,
            'best_val_loss': best_val,
            'total_train_time_s': round(total_train_time, 2),
            'total_val_time_s': round(total_val_time, 2),
            'accuracy': final_metrics['accuracy'],
            'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)
        }
        summary_file = 'results_summary.csv'
        header = not os.path.exists(summary_file)
        pd.DataFrame([summary]).to_csv(summary_file, index=False, mode='a', header=header)
        print(f"Appended summary to {summary_file}")

        # Save validation predictions if primary_key is specified
        if primary_key is not None:
            # check the key exists in both splits
            for split_df in (train_df, val_df):
                if primary_key not in split_df.columns:
                    raise ValueError(f"primary_key '{primary_key}' not found in DataFrame.")

            all_preds = []
            self.model.eval()
            
            def collect_preds(df, loader, factor_name):
                preds = []
                with torch.no_grad():
                    for batch in loader:
                        ids, masks = batch[0].to(self.device), batch[1].to(self.device)
                        out = self.model(ids, attention_mask=masks)
                        p = torch.argmax(out.logits, dim=1)
                        preds.extend(p.cpu().numpy())
                return pd.DataFrame({
                    primary_key: df[primary_key].tolist(),
                    'predicted': preds,
                    'true': df[label_column].tolist(),
                    'split': factor_name
                })

            all_preds.append(collect_preds(train_df, train_loader, 'train'))
            all_preds.append(collect_preds(val_df,   val_loader,   'validation'))

            combined_df = pd.concat(all_preds, ignore_index=True)
            pred_file = f"{cfg_str}_predictions.csv"
            combined_df.to_csv(pred_file, index=False)
            print(f"Saved train+validation predictions to {pred_file}")

        return {
            **final_metrics,
            'train_losses': train_losses,
            'val_losses': val_losses,
            'train_accuracies': train_accs,
            'val_accuracies': val_accs,
            'best_val_loss': best_val,
            'total_train_time': total_train_time,
            'total_val_time': total_val_time
        }

    def evaluate_loss(self, loader, use_amp=True):
        self.model.eval()
        total = 0.0
        true, preds = [], []
        with torch.no_grad():
            for ids, masks, labs in loader:
                ids, masks, labs = ids.to(self.device), masks.to(self.device), labs.to(self.device)
                if use_amp and self.device.type == 'cuda':
                    with autocast():
                        out = self.model(ids, attention_mask=masks, labels=labs)
                else:
                    out = self.model(ids, attention_mask=masks, labels=labs)
                loss = out.loss
                total += loss.item()
                _, p = torch.max(out.logits, dim=1)
                preds.extend(p.cpu().numpy())
                true.extend(labs.cpu().numpy())
        avg = total / len(loader)
        return avg, {
            'true_labels': true,
            'predictions': preds,
            'accuracy': accuracy_score(true, preds)
        }

    def predict(self, texts, primary_key=None, true_labels=None, output_csv=None, max_length=512):
        if isinstance(texts, str):
            texts = [texts]
        if isinstance(texts, pd.Series):
            texts = texts.tolist()

        enc = self.tokenizer(
            texts,
            truncation=True, padding=True,
            max_length=max_length,
            return_tensors='pt'
        )
        ds = TensorDataset(enc['input_ids'], enc['attention_mask'])
        loader = DataLoader(ds, batch_size=self.batch_size)

        preds = []
        self.model.eval()
        for idb, maskb in tqdm(loader, disable=not self.verbose, desc='Predict'):
            idb, maskb = idb.to(self.device), maskb.to(self.device)
            with autocast():
                out = self.model(idb, attention_mask=maskb)
            p = torch.argmax(out.logits, dim=1)
            preds.extend(p.cpu().numpy())
        preds = np.array(preds)

        if output_csv:
            # determine column name & values
            if primary_key is not None and isinstance(primary_key, pd.Series):
                key_name   = primary_key.name
                key_values = primary_key.tolist()
            elif primary_key is not None:
                key_name   = 'id'
                key_values = primary_key
            else:
                key_name   = 'id'
                key_values = np.arange(len(preds))

            # build DataFrame using that key_name
            df = pd.DataFrame({
                key_name:   key_values,
                'predicted': preds
            })
            if true_labels is not None:
                if isinstance(true_labels, pd.Series):
                    true_vals = true_labels.tolist()
                else:
                    true_vals = true_labels
                df['true'] = true_vals

            fname = 'results_predictions.csv'
            df.to_csv(fname, index=False)
            print(f"Saved predictions to {fname}")

        return preds

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)

    def load_model(self, path):
        state = torch.load(path, map_location=self.device)
        self.model.load_state_dict(state)
        self.model.to(self.device)
        self.unfreeze_classifier_layer()
        self.num_unfrozen_bert_layers = self.count_unfrozen_bert_layers()
        print('Loaded model')

    def fine_tune(self, model_wt_path, dataset, text_column, label_column,
                  save_model_path, num_epochs=100, debug=True,
                  print_every=10, early_stop_patience=10):
        self.load_model(model_wt_path)
        res = self._run_train_epoch(
            dataset, num_epochs=num_epochs, test_split=0.2,
            early_stop_patience=early_stop_patience,
            text_column=text_column, label_column=label_column,
            debug=debug, print_every=print_every, primary_key=None
        )
        self.save_model(save_model_path)
        print(f"Model saved to {save_model_path}")
        return res
