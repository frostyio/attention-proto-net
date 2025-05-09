import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
	def __init__(self, input_dim=128, embed_dim=256, max_strokes=32):
		super().__init__()

		self.mlp = nn.Sequential(
			nn.Linear(input_dim, embed_dim),
			nn.LeakyReLU(),
			nn.Linear(embed_dim, embed_dim)
		)

		self.attn = nn.MultiheadAttention(embed_dim, 1)

	def embed(self, x):
		"""
		x: (# batches, # samples, # of strokes, input_dim)
		"""

		# mlp
		s = self.mlp(x) # (batches, samples, strokes, embed_dim)

		# attention
		# expecting (max_strokes "seq_len", batch_size, embed_dim)
		s = s.view(-1, s.shape[2], s.shape[3])
		s = s.transpose(0, 1)

		attn_output, _ = self.attn(s, s, s)

		# aggregating
		embedding = attn_output.mean(dim=0)

		return embedding


	def compute_prototypes(self, support_embeddings, support_labels, n_way):
		prototypes = []

		unique_classes = support_labels.unique()
		class_to_idx = {int(cls): idx for idx, cls in enumerate(unique_classes)}
		remapped_labels = torch.tensor([class_to_idx[int(l.item())] for l in support_labels])

		for cls in range(n_way):
			mask = remapped_labels == cls
			if not mask.any():
				raise ValueError(f"no support examples found for class {cls}")
			class_embeddings = support_embeddings[mask]

			prototype = class_embeddings.mean(dim=0)
			prototypes.append(prototype)
		return torch.stack(prototypes)

	def compute_distances(self, query_embeddings, prototypes):
		return torch.cdist(query_embeddings, prototypes, p=2)

	def forward(self, support_x, support_y, query_x, n_way):
		support_emb = self.embed(support_x)
		query_emb = self.embed(query_x)
		support_y = support_y.squeeze(0)

		prototypes = self.compute_prototypes(support_emb, support_y, n_way)
		dists = self.compute_distances(query_emb, prototypes)

		logits = -dists # (num_query, n_way)
		return logits
