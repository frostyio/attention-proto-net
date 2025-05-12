from typing import Any, TypedDict, Tuple, Optional
from data import DrawingFewShotDataset
from torch.utils.data import DataLoader
from model import Model
from solver import Solver

class DataConfig(TypedDict):
	k_shot: int
	q_queries: int
	episodes_per_epoch: Optional[int]

class ModelConfig(TypedDict):
	d_points: int
	seq_len: int

	n_way: int

	embed_dim: int
	dropout: Tuple[int, int, int]

class OptimizerConfig(TypedDict):
	learning_rate: float
	betas: Tuple[float, float]
	weight_decay: float

class Trainer:
	model: Model
	solver: Solver

	def __init__(
		self,
		training_data: Any,
		validation_data: Any,
		testing_data: Any,

		model_config: ModelConfig,
		optimizer_config: OptimizerConfig,

		training_data_config: DataConfig,
		validation_data_config: DataConfig,
		testing_data_config: DataConfig,


		device="cpu"
	):
		self.training_data = training_data
		self.validation_data = validation_data
		self.testing_data = testing_data

		self.model_config = model_config
		self.optimizer_config = optimizer_config

		self.training_data_config = training_data_config
		self.validation_data_config = validation_data_config
		self.testing_data_config = testing_data_config

		self.device = device

		self.update_data()
		self.update()

	def update_training_data(self):
		self.training_dataset = DrawingFewShotDataset(
			self.training_data,
			self.model_config['n_way'], self.training_data_config['k_shot'], self.training_data_config['q_queries'],
			episodes_per_epoch=self.training_data_config["episodes_per_epoch"]
		)
		self.training_dataloader = DataLoader(self.training_dataset, batch_size=1, shuffle=True)

	def update_validation_data(self):
		self.validation_dataset = DrawingFewShotDataset(
			self.validation_data,
			self.model_config['n_way'], self.validation_data_config['k_shot'], self.validation_data_config['q_queries'],
			episodes_per_epoch=self.validation_data_config["episodes_per_epoch"]
		)
		self.validation_dataloader = DataLoader(self.validation_dataset, batch_size=1, shuffle=True)

	def update_testing_data(self):
		self.testing_dataset = DrawingFewShotDataset(
			self.testing_data,
			self.model_config['n_way'], self.testing_data_config['k_shot'], self.testing_data_config['q_queries'],
			episodes_per_epoch=self.testing_data_config["episodes_per_epoch"]
		)
		self.testing_dataloader = DataLoader(self.testing_dataset, batch_size=1, shuffle=True)

	def update_data(self):
		self.update_training_data()
		self.update_validation_data()
		self.update_testing_data()

	def update(self):
		model_config = self.model_config

		self.model = Model(
			input_dim=model_config["d_points"]*2,
			embed_dim=model_config["embed_dim"],
			max_strokes=model_config["seq_len"],
			dropout=model_config["dropout"]
		).to(self.device)

		self.solver = Solver(
			self.model,
			n_way=model_config["n_way"],
			lr=self.optimizer_config["learning_rate"],
			betas=self.optimizer_config["betas"],
			device=self.device
		)

	def set_model_config(self, config: ModelConfig):
		self.model_config = config
		self.update()

	def set_optimizer_config(self, config: OptimizerConfig):
		self.optimizer_config = config
		self.update()
