from __future__ import annotations

from typing import Iterable, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

try:
	from .utils import preprocess_image
except ImportError:
	from utils import preprocess_image


class EvalDataset(Dataset):
	def __init__(self, df, feature_cols: Iterable[str], image_size: Tuple[int, int] = (224, 224)):
		self.image_paths = df["image_path"].tolist()
		self.structured = df[list(feature_cols)].to_numpy(dtype=np.float32)
		self.has_label = "label" in df.columns
		self.has_thickness = "thickness" in df.columns
		self.labels = df["label"].to_numpy(dtype=np.float32) if self.has_label else None
		self.thickness = df["thickness"].to_numpy(dtype=np.float32) if self.has_thickness else None
		self.image_size = image_size

	def __len__(self):
		return len(self.image_paths)

	def __getitem__(self, idx):
		img = preprocess_image(self.image_paths[idx], size=self.image_size)
		img = torch.from_numpy(np.transpose(img, (2, 0, 1)))
		tab = torch.from_numpy(self.structured[idx])
		out = {"img": img, "tab": tab}
		if self.has_label:
			out["label"] = torch.tensor(self.labels[idx], dtype=torch.float32)
		if self.has_thickness:
			out["thickness"] = torch.tensor(self.thickness[idx], dtype=torch.float32)
		return out


def evaluate_model(
	model: torch.nn.Module,
	test_df,
	feature_cols: Iterable[str],
	weights_path: str | None = None,
	batch_size: int = 32,
	image_size: Tuple[int, int] = (224, 224),
	device: str | None = None,
	multitask: bool = False,
):
	device = device or ("cuda" if torch.cuda.is_available() else "cpu")
	model = model.to(device)
	if weights_path:
		state = torch.load(weights_path, map_location=device)
		model.load_state_dict(state["model_state"] if isinstance(state, dict) and "model_state" in state else state)

	ds = EvalDataset(test_df, feature_cols, image_size)
	loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

	model.eval()
	probs_all, labels_all = [], []
	reg_all, reg_targets = [], []
	with torch.no_grad():
		for batch in loader:
			imgs = batch["img"].to(device)
			tabs = batch["tab"].to(device)
			outputs = model(imgs, tabs)
			if isinstance(outputs, dict):
				if "label" in batch:
					probs_all.append(outputs["cls"].cpu().numpy())
					labels_all.append(batch["label"].numpy())
				if "thickness" in batch:
					reg_all.append(outputs["reg"].cpu().numpy())
					reg_targets.append(batch["thickness"].numpy())
			else:
				# Assume classification single-head in eval
				probs_all.append(outputs.cpu().numpy())
				labels_all.append(batch["label"].numpy())

	metrics = {}
	if labels_all:
		probs = np.concatenate(probs_all)
		labels_np = np.concatenate(labels_all).astype(int)
		preds = (probs >= 0.5).astype(int)
		acc = accuracy_score(labels_np, preds)
		prec, rec, f1, _ = precision_recall_fscore_support(labels_np, preds, average="binary", zero_division=0)
		cm = confusion_matrix(labels_np, preds)
		metrics.update({
			"accuracy": float(acc),
			"precision": float(prec),
			"recall": float(rec),
			"f1": float(f1),
			"confusion_matrix": cm.tolist(),
		})
	if reg_all:
		pred_reg = np.concatenate(reg_all)
		y_reg = np.concatenate(reg_targets)
		mae = float(np.mean(np.abs(pred_reg - y_reg)))
		metrics.update({"mae": mae})
	print(metrics)
	return metrics


