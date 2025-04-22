#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 23:54:06 2025
@model: Bio_ClinicalBERTClassifier.py
author: Midhun Shyam (mshyam)
"""

import os
import time
import pandas as pd
import torch
from torch.optim import AdamW, Adam, SGD
from torch.cuda.amp import autocast, GradScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
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

        # Tokenizer & model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, clean_up_tokenization_spaces=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

        # Optional extra dropout
        if dropout_prob is not None:
            self.model.config.hidden_dropout_prob = dropout_prob
            self.model.config.attention_probs_dropout_prob = dropout_prob

        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # Optimizer setup
        self.optimizer_class = optimizer_class
        self.optimizer_params = optimizer_params
        self.configure_optimizer()

        # Freeze all except classifier
        self.freeze_model_layers()
        self.unfreeze_classifier_layer()

        # Save initial state
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
        num_layers = len(self.model.bert.encoder.layer)
        if n > num_layers:
            raise ValueError(f"Cannot unfreeze {n} layers; model has only {num_layers} layers.")
        for i in range(num_layers - n, num_layers):
            for param in self.model.bert.encoder.layer[i].parameters():
                param.requires_grad = True

    def check_layer_status(self):
        for name, param in self.model.named_parameters():
            status = 'True' if param.requires_grad else 'False'
            print(f"{name}: requires_grad={status}")

    def dataframe_to_dataloader(self, df, shuffle=True, text_column="TEXT", label_column="LABEL", max_length=512):
        if text_column not in df.columns or label_column not in df.columns:
            raise ValueError(f"DataFrame needs columns '{text_column}' and '{label_column}'")
        df = df.copy()
        df[label_column] = df[label_column].astype(int)

        texts = df[text_column].tolist()
        labels = torch.tensor(df[label_column].tolist(), dtype=torch.long)

        enc = self.tokenizer(
            texts, truncation=True, padding=True,
            max_length=max_length, return_tensors="pt"
        )
        dataset = TensorDataset(enc['input_ids'], enc['attention_mask'], labels)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle)

    def _run_train_epoch(
        self,
        data,
        num_epochs=500,
        test_split=0.2,
        early_stop_patience=20,
        shuffle_train=True,
        text_column="TEXT",
        label_column="LABEL",
        debug=True,
        print_every=1
    ):
        scaler = GradScaler()

        # stratified train/val split
        train_df, val_df = train_test_split(
            data, test_size=test_split, random_state=self.seed,
            stratify=data[label_column]
        )
        if debug:
            print(f"Train shape: {train_df.shape}, Val shape: {val_df.shape}")

        train_loader = self.dataframe_to_dataloader(train_df, shuffle=shuffle_train,
                                                    text_column=text_column, label_column=label_column)
        val_loader   = self.dataframe_to_dataloader(val_df, shuffle=False,
                                                    text_column=text_column, label_column=label_column)

        train_losses, val_losses = [], []
        total_train_time = total_val_time = 0.0
        best_val_loss = float('inf')
        best_weights = None
        no_improve = 0

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.1, patience=3, threshold=0.001, verbose=debug
        )

        for epoch in range(1, num_epochs+1):
            # Training
            self.model.train()
            start_t = time.time()
            epoch_train_loss = 0.0
            for ids, masks, labs in train_loader:
                ids, masks, labs = ids.to(self.device), masks.to(self.device), labs.to(self.device)
                self.optimizer.zero_grad()
                with autocast():
                    out = self.model(ids, attention_mask=masks, labels=labs)
                    loss = out.loss
                scaler.scale(loss).backward()
                scaler.step(self.optimizer)
                scaler.update()
                epoch_train_loss += loss.item()
            train_time = time.time() - start_t
            total_train_time += train_time
            avg_train = epoch_train_loss / len(train_loader)
            train_losses.append(avg_train)

            # Validation
            start_v = time.time()
            val_loss, metrics = self.evaluate_loss(val_loader, use_amp=True)
            val_time = time.time() - start_v
            total_val_time += val_time
            val_losses.append(val_loss)

            scheduler.step(val_loss)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_weights = self.model.state_dict().copy()
                no_improve = 0
            else:
                no_improve += 1

            if debug and epoch % print_every == 0:
                print(f"Epoch {epoch:3d} | Train: {avg_train:.4f} | Val: {val_loss:.4f}")

            if no_improve >= early_stop_patience:
                if debug: print(f"Early stopping at epoch {epoch}")
                break

        # load best weights
        if best_weights is not None:
            self.model.load_state_dict(best_weights)
            if debug: print(f"Loaded best model (val_loss={best_val_loss:.4f})")

        # save per-epoch losses
        # build opt_str with full config
        config_items = [f"{k}={v}" for k, v in self.optimizer_params.items()]
        if self.dropout_prob is not None:
            config_items.append(f"dropout={self.dropout_prob}")
        config_items.extend([f"batch_size={self.batch_size}", f"seed={self.seed}"])
        opt_str = f"{self.optimizer_class.__name__}_" + "_".join(config_items)

        loss_df = pd.DataFrame({
            "epoch": list(range(1, len(train_losses)+1)),
            "train_loss": train_losses,
            "val_loss": val_losses
        })
        loss_csv = f"losses_{opt_str}.csv"
        loss_df.to_csv(loss_csv, index=False)
        print(f"Saved losses to {loss_csv}")

        # summary data
        early_stop_triggered = len(train_losses) < num_epochs
        summary_data = {
            "model_name":              self.model_name,
            "num_labels":              self.num_labels,
            "optimizer":               self.optimizer_class.__name__,
            **self.optimizer_params,
            "dropout_prob":            self.dropout_prob,
            "batch_size":              self.batch_size,
            "seed":                    self.seed,
            "test_split":              test_split,
            "early_stop_patience":     early_stop_patience,
            "early_stop_triggered":    early_stop_triggered,
            "epochs_ran":              len(train_losses),
            "best_val_loss":           best_val_loss,
            "total_train_time_s":      round(total_train_time, 2),
            "total_val_time_s":        round(total_val_time, 2)
        }
        print("Training summary details:", summary_data, flush=True)

        summary_df = pd.DataFrame([summary_data])
        summary_file = "training_summary.csv"
        if not os.path.exists(summary_file):
            summary_df.to_csv(summary_file, index=False)
        else:
            summary_df.to_csv(summary_file, mode="a", header=False, index=False)
        print(f"Appended summary to {summary_file}", flush=True)

        return {**metrics,
                "train_losses": train_losses,
                "val_losses": val_losses,
                "best_val_loss": best_val_loss,
                "total_train_time": total_train_time,
                "total_val_time": total_val_time}

    def evaluate_loss(self, loader, use_amp=True):
        self.model.eval()
        total_loss = 0.0
        true, preds = [], []
        with torch.no_grad():
            for ids, masks, labs in loader:
                ids, masks, labs = ids.to(self.device), masks.to(self.device), labs.to(self.device)
                if use_amp and self.device.type == "cuda":
                    with autocast():
                        out = self.model(ids, attention_mask=masks, labels=labs)
                        loss = out.loss
                else:
                    out = self.model(ids, attention_mask=masks, labels=labs)
                    loss = out.loss
                total_loss += loss.item()
                _, p = torch.max(out.logits, dim=1)
                preds.extend(p.cpu().numpy())
                true.extend(labs.cpu().numpy())
        avg = total_loss / len(loader)
        acc = accuracy_score(true, preds)
        return avg, {"true_labels": true, "predictions": preds, "accuracy": acc}

    def print_classification_report(self, true_labels, predictions,
                                    save_figure=True, figure_path='confusion_matrix.png',
                                    title="Confusion Matrix", cmap="viridis"):
        print(classification_report(true_labels, predictions, zero_division=0))
        cm = confusion_matrix(true_labels, predictions)
        import matplotlib.pyplot as plt
        import seaborn as sns
        plt.figure(figsize=(6,4))
        sns.heatmap(cm, annot=True, fmt="d", cmap=cmap,
                    xticklabels=[f"C{i}" for i in range(self.num_labels)],
                    yticklabels=[f"C{i}" for i in range(self.num_labels)])
        plt.title(title)
        plt.xlabel("Predicted"); plt.ylabel("True")
        if save_figure:
            plt.savefig(figure_path, bbox_inches="tight", dpi=100)
            print(f"Saved confusion matrix to {figure_path}")
        plt.show()

    def predict(self, texts, max_length=512):
        if isinstance(texts, str): texts = [texts]
        if isinstance(texts, pd.Series): texts = texts.tolist()
        enc = self.tokenizer(texts, truncation=True, padding=True,
                             max_length=max_length, return_tensors="pt")
        ds = TensorDataset(enc['input_ids'], enc['attention_mask'])
        loader = DataLoader(ds, batch_size=self.batch_size)
        preds = []
        self.model.eval()
        with torch.no_grad():
            for ids, masks in tqdm(loader, disable=not self.verbose, desc="Predicting"):
                ids, masks = ids.to(self.device), masks.to(self.device)
                with autocast():
                    out = self.model(ids, attention_mask=masks)
                p = torch.argmax(out.logits, dim=1)
                preds.extend(p.cpu().numpy())
        return np.array(preds)

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)

    def load_model(self, path):
        state = torch.load(path, map_location=self.device)
        self.model.load_state_dict(state)
        self.model.to(self.device)
        self.unfreeze_classifier_layer()
        print("Loaded model weights.")

    def fine_tune(self, model_wt_path, dataset, text_column, label_column,
                  save_model_path, num_epochs=100, debug=True,
                  print_every=10, early_stop_patience=10):
        self.load_model(model_wt_path)
        results = self._run_train_epoch(
            data=dataset,
            num_epochs=num_epochs,
            test_split=0.2,
            early_stop_patience=early_stop_patience,
            text_column=text_column,
            label_column=label_column,
            debug=debug,
            print_every=print_every
        )
        self.save_model(save_model_path)
        print(f"Fine‑tuned model saved to {save_model_path}")
        return results
