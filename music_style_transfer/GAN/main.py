from music_style_transfer.GAN.data import ToyData, MelodyDataset, Loader, Dataset, ToyDataV2
from music_style_transfer.GAN import model
from music_style_transfer.GAN import trainer
from .config import get_config
import os
import mxnet as mx

def create_toy_model_configs(data):
    generator_config = model.ModelConfig(
        encoder_config=model.EncoderConfig(
            n_layers=1,
            hidden_dim=64),
        embedding_config=model.EmbeddingConfig(
            input_dim=data.num_tokens(),
            hidden_dim=64,
            mask_zero=True),
        conditional_class_config=model.EmbeddingConfig(
            input_dim=data.num_classes(),
            hidden_dim=64,
            mask_zero=False),
        output_layer_config=model.OutputLayerConfig(
            output_dim=data.num_tokens()),
        class_output_layer_config=model.OutputLayerConfig(
            output_dim=data.num_classes()),
        noise_config=model.NoiseConfig(
            noise_dim=64,
            variance=0.05
        )
    )

    discriminator_config = model.ModelConfig(
        encoder_config=model.EncoderConfig(
            n_layers=1,
            hidden_dim=64),
        embedding_config=model.EmbeddingConfig(
            input_dim=data.num_tokens(),
            hidden_dim=64,
            mask_zero=True),
        conditional_class_config=model.EmbeddingConfig(
            input_dim=data.num_classes(),
            hidden_dim=64,
            mask_zero=False),
        output_layer_config=model.OutputLayerConfig(
            output_dim=1,
            softmax=False),
        class_output_layer_config=None,
        noise_config=None
    )

    return generator_config, discriminator_config


def create_toy_train_config():
    config = trainer.TrainConfig(batch_size=1,
                                 discriminator_update_steps=10,
                                 gradient_penalty=0.0,
                                 sampling_frequency=500,
                                 d_label_smoothing=0.0,
                                 d_optimizer=trainer.OptimizerConfig(
                                     learning_rate=5e-5,
                                     optimizer='rmsprop',
                                     optimizer_params='clip_weights:0.2',
                                 ),
                                 g_optimizer=trainer.OptimizerConfig(
                                     learning_rate=5e-5,
                                     optimizer='rmsprop',
                                     optimizer_params=''
                                 ))
    return config


def create_train_config(args):
    config = trainer.TrainConfig(batch_size=args.batch_size,
                                 discriminator_update_steps=args.discriminator_update_steps,
                                 sampling_frequency=args.sampling_frequency,
                                 d_label_smoothing=args.label_smoothing,
                                 d_optimizer=trainer.OptimizerConfig(
                                     learning_rate=args.d_learning_rate,
                                     optimizer='rmsprop',
                                     optimizer_params = 'clip_weights:0.01,clip_gradient:1.0',
                                 ),
                                 g_optimizer=trainer.OptimizerConfig(
                                     learning_rate=args.g_learning_rate,
                                     optimizer='rmsprop',
                                     optimizer_params='clip_gradient:1.0'
                                 ))
    return config


def create_model_configs(args, dataset: Dataset):
    generator_config = model.ModelConfig(
        encoder_config=model.EncoderConfig(
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
            output_dim=dataset.num_classes()),
        noise_config=model.NoiseConfig(
            noise_dim=args.noise_dim,
            variance=0.01
        )
    )

    discriminator_config = model.ModelConfig(
        encoder_config=model.EncoderConfig(
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
        class_output_layer_config=None,
        noise_config=None
    )

    return generator_config, discriminator_config


def main_toy():
    #dataset = ToyData(1)
    dataset = ToyDataV2(1)

    g_config, d_config = create_toy_model_configs(dataset)

    generator = model.Generator(config=g_config)
    discriminator = model.Discriminator(config=d_config)

    t = trainer.Trainer(config=create_toy_train_config(),
                        context=mx.cpu(),
                        generator=generator,
                        discriminator=discriminator)

    t.fit(dataset=dataset,
          epochs=20000,
          samples_output_path='/tmp/out')

def main():
    args = get_config()

    if args.toy:
        main_toy()
        exit(0)

    loader = Loader(path=args.data,
                    max_sequence_length=args.max_seq_len,
                    slices_per_quarter_note=args.slices_per_quarter_note)

    dataset = MelodyDataset(
        melodies=loader.melodies,
        batch_size=args.batch_size)

    if not os.path.exists(args.out_samples):
        os.makedirs(args.out_samples)

    g_config, d_config = create_model_configs(args, dataset)

    generator = model.Generator(config=g_config)
    discriminator = model.Discriminator(config=d_config)

    t = trainer.Trainer(config=create_train_config(args),
                        context=mx.gpu() if args.gpu else mx.cpu(),
                        generator=generator,
                        discriminator=discriminator)

    t.fit(dataset=dataset,
          epochs=args.epochs,
          samples_output_path=args.out_samples)


if __name__ == '__main__':
    #main_toy()
    #exit(0)
    main()
