import tensorflow as tf
from . import data
from . import model
from . import trainer


def create_toy_model_configs(data):
    encoder_config = model.TransformerConfig(
        n_layers=2,
        model_dim=32,
        hidden_dim=32
    )

    embedding_config = model.EmbeddingConfig(
        input_dim=data.num_tokens(),
        hidden_dim=32
    )

    class_embedding_config = model.EmbeddingConfig(
        input_dim=data.num_classes(),
        hidden_dim=32
    )

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
            softmax=False)
    )

    return generator_config, discriminator_config

def main():
    dataset = data.ToyData()

