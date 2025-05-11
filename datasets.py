from google.cloud import storage
from google.cloud.storage import transfer_manager
import os
from typing import Dict, Optional, List, Any
import json
from random import shuffle
from concurrent.futures import ThreadPoolExecutor, as_completed


def list_blobs_with_prefix(bucket_name, prefix):
	storage_client = storage.Client()
	blobs = storage_client.list_blobs(bucket_name, prefix=prefix)
	return [blob.name for blob in blobs if not blob.name.endswith('/')]

def download_dataset(bucket_name, prefix, destination_directory, blob_names=None, workers=8):
	if not blob_names:
		blob_names = list_blobs_with_prefix(bucket_name, prefix)

	blobs_to_download = []
	for blob_name in blob_names:
		local_path = os.path.join(destination_directory, blob_name)
		if not os.path.exists(local_path):
			blobs_to_download.append(blob_name)
		else:
			print(f"skipping {blob_name} (already exists)")

	if not blobs_to_download:
		print("all files already downloaded")
		return

	bucket = storage.Client().bucket(bucket_name)

	results = transfer_manager.download_many_to_path(
		bucket, blobs_to_download, destination_directory=destination_directory, max_workers=workers
	)

	for name, result in zip(blobs_to_download, results):
		if isinstance(result, Exception):
			print(f"failed to download {name}: {result}")
		else:
			print(f"downloaded {name} successfully")

def parse_ndjson(file: str, max_items=None, item_start=0) -> Optional[List[Any]]:
	if not os.path.exists(file):
		print(f"{file} does not exist")
		return None

	if item_start == None:
		item_start = 0

	try:
		data = []
		with open(file, "r", encoding="utf-8") as f:
			lines = f.readlines()
			if max_items != None:
				shuffle(lines)
				lines = lines[item_start:item_start+max_items]

			for line in lines:
				if line.strip():
					item = json.loads(line)
					if isinstance(item, dict):
						data.append(item)
		return data
	except (json.JSONDecodeError, IOError) as e:
		print(f"error parsing ndjson file: {e}")
		return None

def process_blob(source_directory, blob_name, items_per_class, item_offset=0):
	print("processing %s" % blob_name)
	local_path = os.path.join(source_directory, blob_name)
	if not os.path.exists(local_path):
		print(f"failed to parse {blob_name} (does not exist)")
		return None
	name = os.path.basename(blob_name).split('.')[0]
	parsed = parse_ndjson(local_path, max_items=items_per_class, item_start=item_offset)
	return (name, parsed)

def parse_bucket(bucket_name: str, prefix: str, source_directory: str, blob_names=None, max_classes=None, items_per_class=None, item_offset=0):
	if not blob_names:
		blob_names = list_blobs_with_prefix(bucket_name, prefix)

	if max_classes is not None:
		# shuffle(blob_names)
		blob_names = blob_names[:max_classes]

	blobs = {}
	with ThreadPoolExecutor() as executor:
		futures = {executor.submit(process_blob, source_directory, blob_name, items_per_class, item_offset): blob_name for blob_name in blob_names}
		for future in as_completed(futures):
			result = future.result()
			if result:
				name, parsed = result
				blobs[name] = parsed

	return blobs

quickdraw_info = ["quickdraw_dataset", "full/simplified/"]

def blob_names_for_quickdraw_simplified_dataset():
	bucket_name, prefix = quickdraw_info
	return list_blobs_with_prefix(bucket_name, prefix)

def download_quickdraw_simplified_dataset(blob_names=None, destination_directory="../datasets/quickdraw_data", workers=8):
	bucket_name, prefix = quickdraw_info
	download_dataset(bucket_name, prefix, destination_directory, blob_names=blob_names, workers=workers)

def parse_quickdraw_simplified_dataset(source_directory="../datasets/quickdraw_data", blob_names=None, max_classes=None, items_per_class=None, item_offset=0):
	bucket_name, prefix = quickdraw_info
	return parse_bucket(bucket_name, prefix, source_directory, blob_names, max_classes=max_classes, items_per_class=items_per_class, item_offset=item_offset)
