import time
import numpy as np
import os.path as osp
import datetime
from collections import OrderedDict
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from scipy.special import softmax

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.utils import (
    MetricMeter, AverageMeter, tolist_if_not, count_num_param, load_checkpoint,
    save_checkpoint, mkdir_if_missing, resume_from_checkpoint,
    load_pretrained_weights
)

@TRAINER_REGISTRY.register()
class VLBaseLearner(TrainerX):
    """A base trainer for vision language tuning and calibration"""

    def after_train(self):
        print("Finish training")

        print("Testing")
        do_test = not self.cfg.TEST.NO_TEST
        if do_test:
            if self.cfg.TEST.FINAL_MODEL == "best_val":
                print("Deploy the model with the best val performance")
                self.load_model(self.output_dir)
            else:
                print("Deploy the last-epoch model")
            self.test()

        # Show elapsed time
        elapsed = round(time.time() - self.time_start)
        elapsed = str(datetime.timedelta(seconds=elapsed))
        print(f"Elapsed: {elapsed}")

        # Close writer
        self.close_writer()    

    @torch.no_grad()
    def test(self, split=None):
        """A generic testing pipeline."""
        self.set_model_mode("eval")
        self.evaluator.reset()
        
        # prepare the dataset
        if split is None:
            split = self.cfg.TEST.SPLIT

        if split == "val" and self.val_loader is not None:
            data_loader = self.val_loader
        else:
            split = "test"  # in case val_loader is None
            data_loader = self.test_loader 

        print(f"Evaluate on the *{split}* set")

        # Calculate the output
        for batch_idx, batch in enumerate(tqdm(data_loader)):
            input, label = self.parse_batch_test(batch)
            output, image_features, text_features = self.model_inference(input)
            self.evaluator.process(output, label, image_features, text_features)

        # Get logits and labels from evaluator
        logits = np.array(self.evaluator._y_score)
        labels = np.array(self.evaluator._y_true)

        # Apply softmax to get probabilities
        probs = softmax(logits, axis=1)

        # IMPORTANT: Pass None for text_proximity to avoid the error
        text_proximity = None
        
        # Evaluate and log results
        results = self.evaluator.evaluate(probs, labels, text_proximity)

        for k, v in results.items():
            tag = f"{split}/{k}"
            self.write_scalar(tag, v, self.epoch)

        return list(results.values())[0]

    def parse_batch_train(self, batch):
        input = batch["img"]
        label = batch["label"]
        domain = batch["domain"]

        input = input.to(self.device)
        label = label.to(self.device)
        domain = domain.to(self.device)

        return input, label, domain

    def count_unique_labels(self, dataloader):
        unique_labels = set()

        for batch_idx, batch in enumerate(tqdm(dataloader)):
            input, label = self.parse_batch_test(batch)
            unique_labels.update(label.cpu().numpy().tolist())
        print(f"There are {len(unique_labels)} unique labels in the DataLoader.")