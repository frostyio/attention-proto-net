import torch

def save_checkpoint(model, optimizer, epoch, loss, filepath):
	checkpoint = {
		'epoch': epoch,
		'model_state_dict': model.state_dict(),
		'optimizer_state_dict': optimizer.state_dict(),
		'loss': loss
	}
	torch.save(checkpoint, filepath)
	print(f"saved checkpoint to {filepath} (epoch {epoch + 1})")

def load_checkpoint(filepath, model, optimizer=None):
	checkpoint = torch.load(filepath)
	model.load_state_dict(checkpoint['model_state_dict'])
	if optimizer:
		optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

	print(f"loaded checkpoint from {filepath} (epoch {checkpoint['epoch'] + 1})")
	return checkpoint
