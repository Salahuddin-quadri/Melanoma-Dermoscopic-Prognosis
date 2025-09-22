from __future__ import annotations

from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, mean_absolute_error
from tqdm import tqdm

# Support running as module or script
try:
	from .utils import preprocess_image, build_image_augment
except ImportError:
	from utils import preprocess_image, build_image_augment


class FrameDataset(Dataset):
	def __init__(self, df, feature_cols: Iterable[str], image_size: Tuple[int, int] = (224, 224), augment: bool = False):
		self.image_paths = df["image_path"].tolist()
		self.structured = df[list(feature_cols)].to_numpy(dtype=np.float32)
		# Targets for multitask: expect both columns if present
		self.y_cls = df["label"].to_numpy(dtype=np.float32) if "label" in df.columns else None
		self.y_reg = df["thickness"].to_numpy(dtype=np.float32) if "thickness" in df.columns else None
		# Unified target for single-task
		self.y_single = df["target"].to_numpy(dtype=np.float32) if "target" in df.columns else None
		self.image_size = image_size
		self.augment_fn = build_image_augment() if augment else None

	def __len__(self):
		return len(self.image_paths)

	def __getitem__(self, idx):
		img = preprocess_image(self.image_paths[idx], size=self.image_size)
		if self.augment_fn is not None:
			img = self.augment_fn(img)
		# to CHW tensor
		img = torch.from_numpy(np.transpose(img, (2, 0, 1)))
		tab = torch.from_numpy(self.structured[idx])
		out = {"img": img, "tab": tab}
		if self.y_single is not None:
			out["target"] = torch.tensor(self.y_single[idx], dtype=torch.float32)
		if self.y_cls is not None:
			# Clamp to [0,1] for BCE safety
			val = float(self.y_cls[idx])
			val = 1.0 if val > 1 else (0.0 if val < 0 else val)
			out["label"] = torch.tensor(val, dtype=torch.float32)
		if self.y_reg is not None:
			out["thickness"] = torch.tensor(self.y_reg[idx], dtype=torch.float32)
		return out


def train_model(
	model: torch.nn.Module,
	train_df,
	val_df,
	feature_cols: Iterable[str],
	batch_size: int = 32,
	epochs: int = 20,
	image_size: Tuple[int, int] = (224, 224),
	output_dir: str = "outputs",
	device: str | None = None,
	task: str = "classification",
    multitask: bool = False,
):
	Path(output_dir, "checkpoints").mkdir(parents=True, exist_ok=True)
	ckpt_path = str(Path(output_dir, "checkpoints", "best.pt"))

	device = device or ("cuda" if torch.cuda.is_available() else "cpu")
	model = model.to(device)

	train_ds = FrameDataset(train_df, feature_cols, image_size, augment=True)
	val_ds = FrameDataset(val_df, feature_cols, image_size, augment=False)

	def collate(batch):
		imgs = torch.stack([b["img"] for b in batch])
		tabs = torch.stack([b["tab"] for b in batch])
		out = {"imgs": imgs, "tabs": tabs}
		if batch[0].get("target") is not None:
			out["target"] = torch.stack([b["target"] for b in batch])
		if batch[0].get("label") is not None:
			out["label"] = torch.stack([b["label"] for b in batch])
		if batch[0].get("thickness") is not None:
			out["thickness"] = torch.stack([b["thickness"] for b in batch])
		return out

	# Windows multiprocessing can fail to pickle local callables; keep workers=0 for safety
	train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True, collate_fn=collate)
	val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True, collate_fn=collate)

	criterion_cls = torch.nn.BCELoss()
	criterion_reg = torch.nn.L1Loss()
	optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
	best_val_auc = -1.0
	patience, bad_epochs = 5, 0

	for epoch in range(1, epochs + 1):
		model.train()
		running_loss = 0.0
		for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [train]"):
			imgs = batch["imgs"].to(device)
			tabs = batch["tabs"].to(device)
			optimizer.zero_grad()
			outputs = model(imgs, tabs)
			if multitask:
				loss_cls = criterion_cls(outputs["cls"], batch["label"].to(device))
				loss_reg = criterion_reg(outputs["reg"], batch["thickness"].to(device))
				loss = loss_cls + loss_reg
			else:
				targets = batch["target"].to(device)
				if task == "classification":
					loss = criterion_cls(outputs, targets)
				else:
					loss = criterion_reg(outputs, targets)
			loss.backward()
			optimizer.step()
			running_loss += loss.item() * imgs.size(0)
		train_loss = running_loss / len(train_loader.dataset)

		# Validation
		model.eval()
		val_outputs, val_targets = [], []
		with torch.no_grad():
			for batch in tqdm(val_loader, desc=f"Epoch {epoch}/{epochs} [val]"):
				imgs = batch["imgs"].to(device)
				tabs = batch["tabs"].to(device)
				outputs = model(imgs, tabs)
				if multitask:
					val_outputs.append({
						"cls": outputs["cls"].cpu().numpy(),
						"reg": outputs["reg"].cpu().numpy(),
					})
					val_targets.append({
						"cls": batch["label"].numpy(),
						"reg": batch["thickness"].numpy(),
					})
				else:
					val_outputs.append(outputs.cpu().numpy())
					val_targets.append(batch["target"].numpy())
		if multitask:
			probs = np.concatenate([o["cls"] for o in val_outputs])
			y_cls = np.concatenate([t["cls"] for t in val_targets])
			pred_reg = np.concatenate([o["reg"] for o in val_outputs])
			y_reg = np.concatenate([t["reg"] for t in val_targets])
			# AUC may be undefined if only one class in y_true
			try:
				val_auc = roc_auc_score(y_cls, probs)
			except Exception:
				preds = (probs >= 0.5).astype(int)
				val_auc = (preds == y_cls.astype(int)).mean()
			val_mae = mean_absolute_error(y_reg, pred_reg)
			print({"epoch": epoch, "train_loss": train_loss, "val_auc": float(val_auc), "val_mae": float(val_mae)})
			val_metric = val_auc - val_mae  # simple combined objective
			is_better = (best_val_auc < 0) or (val_metric > best_val_auc)
		else:
			val_outputs = np.concatenate(val_outputs)
			val_targets = np.concatenate(val_targets)
			if task == "classification":
				try:
					val_metric = roc_auc_score(val_targets, val_outputs)
				except Exception:
					preds = (val_outputs >= 0.5).astype(int)
					val_metric = (preds == val_targets.astype(int)).mean()
				print({"epoch": epoch, "train_loss": train_loss, "val_auc_or_acc": float(val_metric)})
				is_better = val_metric > best_val_auc
			else:
				val_metric = mean_absolute_error(val_targets, val_outputs)
				print({"epoch": epoch, "train_loss": train_loss, "val_mae": float(val_metric)})
				is_better = (best_val_auc < 0) or (val_metric < best_val_auc)

		# # Early stopping
		# if is_better:
		# 	best_val_auc = val_metric
		# 	bad_epochs = 0
		# 	torch.save({"model_state": model.state_dict()}, ckpt_path)
		# 	# Also save a copy at project root as best.pt
		# 	try:
		# 		root_save = str(Path("best.pt").resolve())
		# 		torch.save({"model_state": model.state_dict()}, root_save)
		# 		print(f"Saved copy to {root_save}")
		# 	except Exception as e:
		# 		print(f"Warning: failed to save root copy: {e}")
		# 	print(f"Saved new best to {ckpt_path}")
		# else:
		# 	bad_epochs += 1
		# 	if bad_epochs >= patience:
		# 		print("Early stopping")
		# 		break

	return {"best_val_auc": float(best_val_auc)}, ckpt_path


