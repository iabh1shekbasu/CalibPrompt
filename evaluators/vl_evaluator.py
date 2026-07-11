import numpy as np
import os.path as osp
from collections import OrderedDict, defaultdict
import torch
from sklearn.metrics import f1_score, confusion_matrix
from dassl.evaluation.build import EVALUATOR_REGISTRY
from dassl.evaluation.evaluator import Classification

from tools.metrics import ECE, MCE, AdaptiveECE, PIECE, ECE_KDE
from tools.plot import plot_reliability_diagram

@EVALUATOR_REGISTRY.register()
class VLClassification(Classification):
    """Evaluator for Vision-Language models."""

    def __init__(self, cfg, lab2cname=None, **kwargs):
        super().__init__(cfg)
        self._lab2cname = lab2cname
        self._correct = 0
        self._total = 0
        self._per_class_res = None
        self._y_score = []
        self._y_true = []
        self._y_pred = []
        if cfg.TEST.PER_CLASS_RESULT:
            assert lab2cname is not None
            self._per_class_res = defaultdict(list)

    def reset(self):
        self._correct = 0
        self._total = 0
        self._y_score = []
        self._y_true = []
        self._y_pred = []
        self._text_features = []
        self._image_features = []
        if self._per_class_res is not None:
            self._per_class_res = defaultdict(list)



    def process(self, mo, gt, image_features, text_features):
        # mo (torch.Tensor): model output [batch, num_classes]
        # gt (torch.LongTensor): ground truth [batch]
        # pred = mo.max(1)[1]
        # matches = pred.eq(gt).float()
        # self._correct += int(matches.sum().item())
        # self._total += gt.shape[0]
        self._y_score.extend(mo.data.cpu().numpy().tolist())
        self._y_true.extend(gt.data.cpu().numpy().tolist())
        # self._y_pred.extend(pred.data.cpu().numpy().tolist())
        self._text_features.extend(text_features.data.cpu().numpy().tolist()) # record text feature and image features in CLIP
        self._image_features.extend(image_features.data.cpu().numpy().tolist())

        # if self._per_class_res is not None:
        #     for i, label in enumerate(gt):
        #         label = label.item()
        #         matches_i = int(matches[i].item())
        #         self._per_class_res[label].append(matches_i)

    def evaluate(self, probs, labels, text_proximity):

        results = OrderedDict()
        ece_bin = self.cfg.CALIBRATION.METRICS.ECE_BINS
        # piece_bin = self.cfg.CALIBRATION.METRICS.PIECE_BINS

        # def debug_ECE(conf, pred, gt, conf_bin_num=10, description=""):
        #     """Debug version of ECE calculation"""
        #     print(f"\nDebugging ECE calculation for: {description}")
        #     print(f"Total samples: {len(conf)}")
        #     print(f"Average confidence: {np.mean(conf):.4f}")
        #     print(f"Accuracy: {100*np.mean(pred == gt):.2f}%")
            
        #     bins = np.linspace(0, 1, conf_bin_num+1)
        #     bin_indices = np.digitize(conf, bins) - 1
            
        #     print("\nBin Analysis:")
        #     print(f"{'Bin Range':<15} {'Samples':<10} {'Accuracy':<10} {'Conf':<10} {'|Acc-Conf|':<10} {'Weight':<10} {'Contribution':<10}")
        #     print("-" * 75)
            
        #     bin_acc = []
        #     bin_confidences = []
        #     total_contribution = 0
            
        #     for i in range(conf_bin_num):
        #         in_bin = bin_indices == i
        #         n_in_bin = np.sum(in_bin)
                
        #         if n_in_bin > 0:
        #             accuracy = np.mean(gt[in_bin] == pred[in_bin])
        #             mean_confidence = np.mean(conf[in_bin])
        #         else:
        #             accuracy = 0
        #             mean_confidence = 0
                    
        #         bin_acc.append(accuracy)
        #         bin_confidences.append(mean_confidence)
                
        #         weight = n_in_bin / len(conf)
        #         contribution = weight * abs(accuracy - mean_confidence)
        #         total_contribution += contribution
                
        #         bin_range = f"{bins[i]:.2f}-{bins[i+1]:.2f}"
        #         print(f"{bin_range:<15} {n_in_bin:<10d} {accuracy*100:>7.2f}% {mean_confidence*100:>8.2f}% {abs(accuracy-mean_confidence)*100:>9.2f}% {weight:>9.4f} {contribution:>10.4f}")
            
        #     print(f"\nTotal ECE: {total_contribution*100:.2f}%")
        #     return total_contribution
        total = len(labels)

        # make the prediction
        preds = np.argmax(probs, axis=1)

        correct = np.sum(preds == labels)

        accuracy = 100.0 * correct / total

        error = 100.0 - accuracy

        macro_f1 = 100.0 * f1_score(
            labels,
            preds,
            average="macro",
            labels=np.unique(labels)
        )

        confs = probs[range(probs.shape[0]), preds]
        avg_conf = np.mean(confs)

        ece = 100.0 * ECE(confs, preds, labels, ece_bin) # debug

        mce = 100.0 * MCE(confs, preds, labels, ece_bin)

        ace = 100.0 * AdaptiveECE(confs, preds, labels, ece_bin)

        ece_kde = 100.0 * ECE_KDE(confs, preds, labels, p=1)

        # piece = 0.0 # or 0.0 if you prefer
        # if text_proximity is not None:
        #     piece = 100.0 * PIECE(confs, text_proximity, preds, labels, piece_bin, ece_bin)

        # Calculate ECE-KDE
        # In vl_evaluator.py
        # Calculate ECE-KDE
        print(f"Debug - probs shape: {probs.shape}, dtype: {probs.dtype}")
        print(f"Debug - preds shape: {preds.shape}, dtype: {preds.dtype}")
        print(f"Debug - labels shape: {labels.shape}, dtype: {labels.dtype}")

        # The first value will be returned by trainer.test()
        results["accuracy"] = accuracy
        results["error_rate"] = error
        results["macro_f1"] = macro_f1
        results["confidence"] = avg_conf
        results["ece"] = ece
        results["mce"] = mce
        results["ace"] = ace
        # results["piece"] = piece
        results["ece_kde"] = ece_kde

        # Calculate per-class metrics
        unique_labels = np.unique(labels)
        
        # Ground Truth Based ECE
        # This calculates "For samples that are actually class X (ground truth), what percentage did we get right?"
        print("\n=> Per-class ECE (Ground Truth Based)")
        print(f"{'Class':<15} {'Samples':<10} {'Accuracy':<10} {'ECE':<10}")
        print("-" * 45)
        
        gt_ece_values = []
        for label in unique_labels:
            # Get indices where true label is this class
            gt_indices = labels == label
            gt_confs = confs[gt_indices]
            gt_preds = preds[gt_indices]
            gt_labels = labels[gt_indices]
            
            n_samples = np.sum(gt_indices)
            class_acc = 100.0 * np.mean(gt_preds == gt_labels)
            class_ece = 100.0 * ECE(gt_confs, gt_preds, gt_labels, ece_bin) #debug
            gt_ece_values.append(class_ece)
            
            label_name = self._lab2cname.get(label, f"Label_{label}")
            print(f"{label_name:<15} {n_samples:<10d} {class_acc:>7.2f}%  {class_ece:>7.2f}%")
            
            # Store results
            results[f"gt_ece_class_{label_name}"] = float(class_ece)
            results[f"gt_acc_class_{label_name}"] = float(class_acc)
            results[f"gt_samples_class_{label_name}"] = int(n_samples)
        
        # Prediction Based ECE
        # This calculates "For samples we predicted as class X, what percentage did we get right?" (precision)
        print("\n=> Per-class ECE (Prediction Based)")
        print(f"{'Class':<15} {'Predictions':<12} {'Precision':<10} {'ECE':<10}")
        print("-" * 47)
        
        pred_ece_values = []
        for label in unique_labels:
            # Get indices where model predicted this class
            pred_indices = preds == label
            pred_confs = confs[pred_indices]
            pred_preds = preds[pred_indices]
            pred_labels = labels[pred_indices]
            
            n_predictions = np.sum(pred_indices)
            class_precision = 100.0 * np.mean(pred_preds == pred_labels)
            class_ece = 100.0 * ECE(pred_confs, pred_preds, pred_labels, ece_bin) # debug
            pred_ece_values.append(class_ece)
            
            label_name = self._lab2cname.get(label, f"Label_{label}")
            print(f"{label_name:<15} {n_predictions:<12d} {class_precision:>7.2f}%  {class_ece:>7.2f}%")
            
            # Store results
            results[f"pred_ece_class_{label_name}"] = float(class_ece)
            results[f"pred_precision_class_{label_name}"] = float(class_precision)
            results[f"pred_samples_class_{label_name}"] = int(n_predictions)
        
        # Store mean ECE values
        results["mean_gt_ece"] = float(np.mean(gt_ece_values))
        results["mean_pred_ece"] = float(np.mean(pred_ece_values))
        
        print(f"\nMean Ground Truth Based ECE: {results['mean_gt_ece']:.2f}%")
        print(f"Mean Prediction Based ECE: {results['mean_pred_ece']:.2f}%")


        print(
            "=> result\n"
            f"* total: {total:,}\n"
            f"* correct: {correct:,}\n"
            f"* accuracy: {accuracy:.2f}%\n"
            f"* error: {error:.2f}%\n"
            f"* macro_f1: {macro_f1:.2f}%\n"
            f"* confidence: {avg_conf:.2f}%\n"
            f"* ece: {ece:.2f}%\n"
            f"* mce: {mce:.2f}%\n"
            f"* ace: {ace:.2f}%\n"
            # f"* piece: {piece:.2f}%\n"
            f"* ece_kde: {ece_kde:.2f}%\n"
        )
        
        # plot ece
        base_dir = self.cfg.OUTPUT_DIR
        base_name = self.cfg.DATASET.NAME + '_' + self.cfg.TRAINER.NAME

        # if self.cfg.CALIBRATION.SCALING.IF_SCALING:
        #     base_name = base_name + '_' + str(self.cfg.CALIBRATION.SCALING.MODE)

        base_name  = base_name + '_overall_ece.png'
        plot_dir = osp.join(base_dir, base_name)

        plot_reliability_diagram(preds, confs, labels, ece_bin, None, plot_dir)

        # Plot per-class reliability diagrams for ground-truth based
        for label in unique_labels:
            label_name = self._lab2cname.get(label, f"Label_{label}")
            gt_indices = labels == label
            plot_name = base_name.replace('_overall_ece.png', f'_gt_based_{label_name}_ece.png')
            plot_dir = osp.join(base_dir, plot_name)
            plot_reliability_diagram(
                preds[gt_indices],
                confs[gt_indices],
                labels[gt_indices],
                ece_bin,
                None,
                plot_dir
            )

        # Plot per-class reliability diagrams for prediction based
        for label in unique_labels:
            label_name = self._lab2cname.get(label, f"Label_{label}")
            pred_indices = preds == label
            plot_name = base_name.replace('_overall_ece.png', f'_pred_based_{label_name}_ece.png')
            plot_dir = osp.join(base_dir, plot_name)
            plot_reliability_diagram(
                preds[pred_indices],
                confs[pred_indices],
                labels[pred_indices],
                ece_bin,
                None,
                plot_dir
            )

        return results