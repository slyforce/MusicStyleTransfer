from music_style_transfer.GAN.data import ToyData, MelodyDataset, Loader, Dataset
from music_style_transfer.GAN import model
from music_style_transfer.GAN import trainer
from .config import get_config
import os
import tensorflow as tf


def create_toy_model_configs(data):
    generator_config = model.ModelConfig(
        encoder_config=model.TransformerConfig(
            n_layers=1,
            hidden_dim=4),
        embedding_config=model.EmbeddingConfig(
            input_dim=data.num_tokens(),
            hidden_dim=4,
            mask_zero=True),
        conditional_class_config=model.EmbeddingConfig(
            input_dim=data.num_classes(),
            hidden_dim=4,
            mask_zero=False),
        output_layer_config=model.OutputLayerConfig(
            output_dim=data.num_tokens()),
        class_output_layer_config=model.OutputLayerConfig(
            output_dim=data.num_classes())
    )

    discriminator_config = model.ModelConfig(
        encoder_config=model.TransformerConfig(
            n_layers=1,
            hidden_dim=4),
        embedding_config=model.EmbeddingConfig(
            input_dim=data.num_tokens(),
            hidden_dim=4,
            mask_zero=True),
        conditional_class_config=model.EmbeddingConfig(
            input_dim=data.num_classes(),
            hidden_dim=4,
            mask_zero=False),
        output_layer_config=model.OutputLayerConfig(
            output_dim=1,
            softmax=False),
        class_output_layer_config=None
    )

    return generator_config, discriminator_config


def create_toy_train_config():
    config = trainer.TrainConfig(batch_size=1,
                                 d_optimizer=trainer.OptimizerConfig(
                                     learning_rate=0.,
                                     optimizer='adam'
                                 ),
                                 g_optimizer=trainer.OptimizerConfig(
                                     learning_rate=0.,
                                     optimizer='adam'
                                 ))
    return config


def create_train_config(args):
    config = trainer.TrainConfig(batch_size=args.batch_size,
                                 d_optimizer=trainer.OptimizerConfig(
                                     learning_rate=args.d_learning_rate,
                                     optimizer='adam'
                                 ),
                                 g_optimizer=trainer.OptimizerConfig(
                                     learning_rate=args.g_learning_rate,
                                     optimizer='adam'
                                 ))
    return config


def create_model_configs(args, dataset: Dataset):
    generator_config = model.ModelConfig(
        encoder_config=model.TransformerConfig(
            n_layers=args.g_n_layers,
            hidden_dim=args.g_rnn_hidden_dim),
        embedding_config=model.EmbeddingConfig(
            input_dim=dataset.num_tokens(),
            hidden_dim=args.g_emb_hidden_dim,
            mask_zero=True),
        conditional_class_config=model.EmbeddingConfig(
            input_dim=dataset.num_classes(),
            hidden_dim=args.g_emb_hidden_dim,
            mask_zero=False),
        output_layer_config=model.OutputLayerConfig(
            output_dim=dataset.num_tokens()),
        class_output_layer_config=model.OutputLayerConfig(
            output_dim=dataset.num_classes())
    )

    discriminator_config = model.ModelConfig(
        encoder_config=model.TransformerConfig(
            n_layers=args.d_n_layers,
            hidden_dim=args.d_rnn_hidden_dim),
        embedding_config=model.EmbeddingConfig(
            input_dim=dataset.num_tokens(),
            hidden_dim=args.d_emb_hidden_dim,
            mask_zero=True),
        conditional_class_config=model.EmbeddingConfig(
            input_dim=dataset.num_classes(),
            hidden_dim=args.d_emb_hidden_dim,
            mask_zero=False),
        output_layer_config=model.OutputLayerConfig(
            output_dim=1,
            softmax=False),
        class_output_layer_config=None
    )

    return generator_config, discriminator_config


def main_toy():
    dataset = ToyData(1)

    g_config, d_config = create_toy_model_configs(dataset)

    generator = model.Generator(config=g_config)
    discriminator = model.Discriminator(config=d_config)

    t = trainer.Trainer(config=create_toy_train_config(),
                        generator=generator,
                        discriminator=discriminator)

    t.fit(dataset=dataset,
          epochs=2)


def main():
    args = get_config()

    loader = Loader(path=args.data,
                    max_sequence_length=args.max_seq_len,
                    slices_per_quarter_note=args.slices_per_quarter_note)

    dataset = MelodyDataset(
        melodies=loader.melodies,
        batch_size=args.batch_size)

    if args.out_samples is not None:
        if not os.stat(args.out_samples):
            os.mkdir(args.out_samples)

    g_config, d_config = create_model_configs(args, dataset)

    generator = model.Generator(config=g_config)
    discriminator = model.Discriminator(config=d_config)

    t = trainer.Trainer(config=create_train_config(args),
                        generator=generator,
                        discriminator=discriminator)

    if args.gpu:
        device = '/gpu:0'
    else:
        device = '/cpu:0'

    with tf.device(device):
        t.fit(dataset=dataset,
              epochs=200,
              samples_output_path=args.out_samples)


if __name__ == '__main__':
    # main_toy()
    # exit(0)
    main()
