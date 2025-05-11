import numpy as np
from pprint import pprint
import torch
from torch.utils.data import Dataset
from torch import tensor
import random

def clean_data(data):
	cleaned_data = []
	for class_arr in data.values():
		recognized_mask = np.array([value["recognized"] for value in class_arr], dtype=bool)
		values = np.array(list(class_arr))
		cleaned_data.append(values[recognized_mask])
	return cleaned_data

def preprocess_data(data, max_strokes=32, max_points=64):
	processed = {}
	for class_arr in data:
		entries = []
		word = ""
		for entry in class_arr:
			drawings = entry["drawing"]
			word = entry["word"]
			strokes = []

			for stroke in drawings:
				x_points, y_points = stroke
				interleaved = np.empty((len(x_points) + len(y_points),), dtype=np.float32)
				interleaved[0::2] = x_points
				interleaved[1::2] = y_points

				if len(interleaved) > max_points * 2: # each point is 2 numbers
					interleaved = interleaved[:max_points * 2]
				else:
					# pad with zeros
					pad_len = max_points * 2 - len(interleaved)
					interleaved = np.pad(interleaved, (0, pad_len))

				strokes.append(interleaved)
				if len(strokes) >= max_strokes:
					break

			while len(strokes) < max_strokes:
				strokes.append(np.zeros(max_points * 2, dtype=np.float32))

			entries.append(np.stack(strokes))

		if word != "":
			processed[word] = entries
		else:
			print("word is some reason blank")
			print(class_arr)

	return processed

class DrawingFewShotDataset(Dataset):
	def __init__(self, processed_data, n_way, k_shot, q_queries, word2idx=None, episodes_per_epoch=None):
		# process_data = {
		#	["word"] = [
		# 		numpy_array (max_strokes, max_points*2)
		# 	]
		# }

		data = []
		labels = []
		self.word2idx = word2idx or {word: i for i, word in enumerate(processed_data.keys())}
		self.idx2word = {i: word for word, i in self.word2idx.items()}

		self.n_way = n_way
		self.k_shot = k_shot
		self.q_queries = q_queries
		self.episodes_per_epoch = episodes_per_epoch

		for word, entry_list in processed_data.items():
			for entry in entry_list:
				data.append(torch.from_numpy(entry).float())
				labels.append(self.word2idx[word])

		self.data = torch.stack(data)
		self.labels = torch.tensor(labels)

	def __len__(self):
		if self.episodes_per_epoch is None:
			return len(self.data)
		else:
			return self.episodes_per_epoch

	def __getitem__(self, idx):
		sampled_classes = random.sample(set(self.labels.tolist()), self.n_way)

		support_set, support_labels = [], []
		query_set, query_labels = [], []

		for cls in sampled_classes:
			class_indices = (self.labels == cls).nonzero(as_tuple=True)[0].tolist()
			selected_indices = random.sample(class_indices, self.k_shot + self.q_queries)
			support_set.extend(selected_indices[: self.k_shot])
			query_set.extend(selected_indices[self.k_shot :])
			support_labels.extend([cls] * self.k_shot)
			query_labels.extend([cls] * self.q_queries)

		support_embeddings = self.data[support_set]
		query_embeddings = self.data[query_set]

		return (
			support_embeddings,
			tensor(support_labels),
			query_embeddings,
			tensor(query_labels)
		)
