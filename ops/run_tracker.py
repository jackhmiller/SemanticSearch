from dataclasses import dataclass
import os
import uuid
from datetime import datetime


@dataclass
class Tracker:
	model: str = None
	embedding_features: list = None
	test_sentence: str = None
	test_item: str = None
	cos_sim: float = None
	run_id: str = None
	date: str = None
	user: str = None

	def __post_init__(self):
		if self.user is None:
			self.user= os.getlogin()
		if self.date is None:
			now = datetime.now()
			self.date = now.strftime("%Y_%m_%d_%H_%M_%S")
			self.user= os.getlogin()
		if self.run_id is None:
			self.run_id= str(uuid.uuid1())