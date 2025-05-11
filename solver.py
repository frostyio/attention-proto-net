from torch import optim, no_grad, argmax, save, load, tensor
import torch.nn.functional as F
import torch.nn as nn
from numpy import mean
from tqdm import tqdm
import csv
import optuna

class Solver:
	def __init__(
		self,
		model,
		n_way=2,
		lr=0.0001,
		betas=(0.9,0.999),
		weight_decay=1e-4,
		device="cpu"
	):
		self.model = model.to(device)
		self.optimizer = optim.Adam(self.model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)
		self.classifier = nn.NLLLoss()
		self.n_way = n_way
		self.device = device
		self.epoch = 0
		self.loss = 0

	def train(self, training_loader, validation_loader=None, trial=None, n_epochs=200):
		prev = self.epoch
		for epoch in range(prev, prev+n_epochs):
			self.model.train()
			epoch_loss = 0.0
			self.epoch = epoch

			for (
				support_x,
				support_y,
				query_x,
				query_y,
			) in training_loader:
				support_x, support_y = support_x.to(self.device), support_y.to(self.device)
				query_x, query_y = query_x.to(self.device), query_y.to(self.device)

				self.optimizer.zero_grad()

				query_y = query_y.squeeze(0)
				unique_classes = query_y.unique()
				class_to_idx = {int(cls): idx for idx, cls in enumerate(unique_classes)}
				query_y = tensor([class_to_idx[int(l.item())] for l in query_y]).to(self.device)

				logits = self.model.forward(support_x, support_y, query_x, self.n_way)

				loss = self.classifier(F.log_softmax(logits, dim=-1), query_y)

				loss.backward()
				self.optimizer.step()
				epoch_loss += loss.item()

			avg_loss = epoch_loss / len(training_loader)
			self.loss = avg_loss
			print(f"epoch {epoch+1}/{prev+n_epochs}, loss: {avg_loss:.4f}, train. acc.: {self.eval(training_loader):.4f}", end="")

			if validation_loader:
				val_acc = self.eval(validation_loader)
				print(f", val. acc.: {val_acc:.4f}")
				if trial is not None:
					# report for pruning
					trial.report(val_acc, step=epoch)

					if trial.should_prune():
						raise optuna.TrialPruned()

	def eval(self, loader):
		device = self.device
		accuracies = []

		self.model.eval()
		with no_grad():
			for (
				support_x,
				support_y,
				query_x,
				query_y,
			) in loader:
				support_x, support_y = support_x.to(self.device), support_y.to(self.device)
				query_x, query_y = query_x.to(self.device), query_y.to(self.device)

				query_y = query_y.squeeze(0)
				unique_classes = query_y.unique()
				class_to_idx = {int(cls): idx for idx, cls in enumerate(unique_classes)}
				query_y = tensor([class_to_idx[int(l.item())] for l in query_y]).to(self.device)

				logits = self.model.forward(support_x, support_y, query_x, self.n_way)
				log_probs = F.log_softmax(logits, dim=-1)
				pred_labels = argmax(log_probs, dim=-1)
				accuracies.append((pred_labels == query_y).float().mean().item())
			# print(f"avg test accuracy: {mean(accuracies) * 100:.2f}%")
		return mean(accuracies)

	def predict(self, support_x, support_y, query_x):
		self.model.eval()
		with no_grad():
			support_x, support_y, query_x = (
				support_x.to(self.device),
				support_y.to(self.device),
				query_x.to(self.device),
			)

			logits = self.model.forward(support_x, support_y, query_x, self.n_way)
			log_probs = F.log_softmax(logits, dim=-1)
			probs = log_probs.exp()
		return probs
