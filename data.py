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


from threading import RLock
from concurrent.futures import ThreadPoolExecutor
from os import path, remove
import json

blobs_cached = {}
blobs_lock = RLock()

def do_preprocessing_for_blobs(blob_name, n_samples: int):
	"""
	process n amount of lines in a blob and cache them
	"""
	global blobs_cached, blobs_lock

	# with blobs_lock:
	blob = blobs_cached.get(blob_name, {"first_samples": [], "word": ""})
	samples = blob["first_samples"]

	# process needed samples
	current_n_samples = len(samples)
	remaining = max(0, n_samples - current_n_samples)
	range_to_process = range(current_n_samples, current_n_samples + n_samples)

	# no more samples needed to process
	if remaining == 0:
		return

	if not path.exists(blob_name):
		print(f"failed to parse {blob_name} (does not exist)")
		return None

	print(f"needing to process {blob_name} with remaining samples {remaining}, so {range_to_process}")

	# process new data
	data = []
	try:
		with open(blob_name, "r", encoding="utf-8") as f:
			lines = f.readlines()[range_to_process.start:range_to_process.stop]

			for i, line in enumerate(lines):
				if line.strip():
					item = json.loads(line)
					if isinstance(item, dict):
						# print(f"processed line nbr {i}")
						data.append(item)
	except (json.JSONDecodeError, IOError) as e:
		print(f"error parsing ndjson file: {e}")

	# messy processing
	data = clean_data({"data": data})
	data = preprocess_data(data)
	if len(data.keys()) <= 0:
		return
	word = next(iter(data.keys()))
	data = data[word]

	# add to cached samples
	# with blobs_lock:
	if blob_name not in blobs_cached:
		blobs_cached[blob_name] = {"first_samples": [], "word": ""}

	blobs_cached[blob_name]["first_samples"] += data
	blobs_cached[blob_name]["word"] = word

def get_class_samples(key, n_samples, idx_offset=0):
	# with blobs_lock:
	cached = blobs_cached[key]
	samples = cached["first_samples"][idx_offset:idx_offset+n_samples]
	return (cached["word"], samples)

def get_preprocessed_data(
	data_dir, data_blobs, # blob = class
	tr_n_way, tr_samples, # training # classes, training # samples per class
	va_n_way, va_samples, # validation # classes, validation # samples per class
	te_n_way, te_samples, # testing # classes, testing # samples per class
):
	# find blobs not preprocessed
	preprocess_blobs = [path.join(data_dir, name) for name in data_blobs if name not in blobs_cached]

	# split
	keys = list(set(list(blobs_cached.keys()) + preprocess_blobs))

	total_n_way = tr_n_way + va_n_way + te_n_way
	if total_n_way > len(keys):
		raise ValueError("not enough classes")

	keys = keys[0:total_n_way] # get only the keys that will be used

	tr_keys = keys[0:tr_n_way]
	va_keys = keys[tr_n_way:tr_n_way+va_n_way]
	te_keys = keys[tr_n_way+va_n_way:total_n_way]

	with ThreadPoolExecutor(max_workers=total_n_way) as executor:
		futures = []
		futures += [executor.submit(do_preprocessing_for_blobs, blob, tr_samples) for blob in tr_keys if blob in preprocess_blobs]
		futures += [executor.submit(do_preprocessing_for_blobs, blob, va_samples) for blob in va_keys if blob in preprocess_blobs]
		futures += [executor.submit(do_preprocessing_for_blobs, blob, te_samples) for blob in te_keys if blob in preprocess_blobs]

		for i, future in enumerate(futures):
			result = future.result()
			print(f"{i + 1} / {total_n_way} done")

	with blobs_lock:
		get_data = lambda k, n_samples: {
			word: samples
			for key in k
			if key in blobs_cached
			for word, samples in [get_class_samples(key, n_samples)]
		}

		tr_data = get_data(tr_keys, tr_samples)
		va_data = get_data(va_keys, va_samples)
		te_data = get_data(te_keys, te_samples)

		return (tr_data, va_data, te_data)
