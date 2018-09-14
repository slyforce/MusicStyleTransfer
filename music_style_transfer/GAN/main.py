import tensorflow as tf
from . import data
from . import model
from . import trainer


def create_toy_model_configs(data):
    generator_config = model.GeneratorConfig(
        encoder_config=model.TransformerConfig(
            n_layers=2,
            model_dim=32,
            hidden_dim=32),
        embedding_config=model.EmbeddingConfig(
            input_dim=data.num_tokens(),
            hidden_dim=32),
        conditional_class_config=model.EmbeddingConfig(
            input_dim=data.num_classes(),
            hidden_dim=32),
        output_layer_config=model.OutputLayerConfig(
            output_dim=data.num_tokens()),
        class_output_layer_config=model.OutputLayerConfig(
            output_dim=data.num_classes())
    )

    discriminator_config = model.GeneratorConfig(
        encoder_config=model.TransformerConfig(
            n_layers=2,
            model_dim=32,
            hidden_dim=32),
        embedding_config=model.EmbeddingConfig(
            input_dim=data.num_tokens(),
            hidden_dim=32),
        conditional_class_config=model.EmbeddingConfig(
            input_dim=data.num_classes(),
            hidden_dim=32),
        output_layer_config=model.OutputLayerConfig(
            output_dim=1,
            softmax=False),
        class_output_layer_config=None
    )

    return generator_config, discriminator_config


def create_toy_train_config():
    config = trainer.TrainConfig(batch_size=1,
                                 d_optimizer=trainer.OptimizerConfig(
                                     learning_rate=1e-4,
                                     optimizer='adam'
                                 ),
                                 g_optimizer=trainer.OptimizerConfig(
                                     learning_rate=1e-4,
                                     optimizer='adam'
                                 ))
    return config


def main_toy():
    dataset = data.ToyData()

    g_config, d_config = create_toy_model_configs(dataset)

    generator = model.Generator(config=g_config)
    discriminator = model.Discriminator(config=d_config)

    t = trainer.Trainer(config=create_toy_train_config(),
                        generator=generator,
                        discriminator=discriminator)

    t.fit(dataset=dataset,
          epochs=10)