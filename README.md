# Stroke-Based Prototypical Network with Attention Pooling

An implementation of a Prototypical Network designed for stroke-based classification for few-shot learning. The model embeds each sample (a set of strokes) using an MLP followed by a single-head self-attention and attention pooling.

### MLP Encoder

Input_dim -> embed_dim -> embed_dim

### Self-Attention

Single-head MultiHeadAttention capturing intra-sample dependencies

### Attention Pooling

The model learns a weighted aggregation of strokes using a trainable scoring layer followed by a softmax

### Prototypical Inference

- Compute class prototypes
- Compute Euclidean distance from query to each prototype
- Return negative distances for classification

## Notes

- Avoids CNNs entirely for experimental purposes
- Avoids cross-attention for future compute constraints

## Data

`data.py` and `datasets.py` are currently setup for [The Quick, Draw!](https://github.com/googlecreativelab/quickdraw-dataset) dataset.
