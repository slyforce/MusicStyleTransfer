import tensorflow as tf
from config import get_config
from trainer import Trainer

from data import Loader, FeatureManager

sess = tf.Session()
config, _ = get_config()

data_loader = Loader(config.dataset, config.restrict)
feature_manager = FeatureManager(data_loader.restrictor)
feature_manager.generate_data(data_loader.melodies)

trainer = Trainer(config, feature_manager)

trainer.train()
