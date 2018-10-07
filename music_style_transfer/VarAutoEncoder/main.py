from music_style_transfer.GAN.data import ToyData, MelodyDataset, Loader, Dataset, ToyDataV2
from music_style_transfer.VarAutoEncoder import model
from music_style_transfer.VarAutoEncoder import trainer
from .config import get_config
import os
import mxnet as mx

def create_toy_model_configs(data):
    decoder_config = model.DecoderConfig(
        encoder_config=model.LSTMConfig(
            n_layers=1,
            hidden_dim=32,
            dropout=0.0),
        output_layer_config=model.OutputLayerConfig(
            output_dim=data.num_tokens()),
        latent_dimension=16,
        input_classes=data.num_classes()
    )

    encoder_config = model.EncoderConfig(
        encoder_config=model.LSTMConfig(
            n_layers=1,
            hidden_dim=32,
            dropout=0.0),
        embedding_config=model.EmbeddingConfig(
            input_dim=data.num_tokens(),
            hidden_dim=8,
            mask_zero=True),
        latent_dimension=16,
        input_classes=data.num_classes()
    )

    return decoder_config, encoder_config


def create_toy_train_config():
    config = trainer.TrainConfig(batch_size=1,
                                 sampling_frequency=500,
                                 checkpoint_frequency=500,
                                 num_checkpoints_not_improved=-1,
                                 optimizer=trainer.OptimizerConfig(
                                     learning_rate=1e-4,
                                     optimizer='adam',
                                     optimizer_params = 'clip_gradient:1.0',
                                 ))
    return config


def create_train_config(args):
    config = trainer.TrainConfig(batch_size=args.batch_size,
                                 sampling_frequency=args.sampling_frequency,
                                 checkpoint_frequency=args.checkpoint_frequency,
                                 num_checkpoints_not_improved=args.num_checkpoints_not_improved,
                                 optimizer=trainer.OptimizerConfig(
                                     learning_rate=args.learning_rate,
                                     optimizer=args.optimizer,
                                     optimizer_params=args.optimizer_params,
                                 ))
    return config


def create_model_configs(args, dataset: Dataset):
    decoder_config = model.DecoderConfig(
        encoder_config=model.LSTMConfig(
            n_layers=args.d_n_layers,
            hidden_dim=args.d_rnn_hidden_dim,
            dropout=args.d_dropout),
        output_layer_config=model.OutputLayerConfig(
            output_dim=dataset.num_tokens()),
        latent_dimension=args.latent_dim,
        input_classes=dataset.num_classes()
    )

    encoder_config = model.EncoderConfig(
        encoder_config=model.LSTMConfig(
            n_layers=args.e_n_layers,
            hidden_dim=args.e_rnn_hidden_dim,
            dropout=args.e_dropout),
        embedding_config=model.EmbeddingConfig(
            input_dim=dataset.num_tokens(),
            hidden_dim=args.e_emb_hidden_dim,
            mask_zero=True),
        latent_dimension=args.latent_dim,
        input_classes=dataset.num_classes()
    )

    return decoder_config, encoder_config


def main_toy():
    dataset = ToyData(1)
    #dataset = ToyDataV2(1)

    d_config, e_config = create_toy_model_configs(dataset)

    decoder = model.Decoder(config=d_config)
    encoder = model.Encoder(config=e_config)

    t = trainer.Trainer(config=create_toy_train_config(),
                        context=mx.cpu(),
                        decoder=decoder,
                        encoder=encoder)

    t.fit(dataset=dataset,
          validation_dataset=dataset,
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

    d_config, e_config = create_model_configs(args, dataset)

    decoder = model.Decoder(config=d_config)
    encoder = model.Encoder(config=e_config)

    t = trainer.Trainer(config=create_train_config(args),
                        context=mx.gpu() if args.gpu else mx.cpu(),
                        decoder=decoder,
                        encoder=encoder)

    t.fit(dataset=dataset,
          validation_dataset=dataset,
          epochs=args.epochs,
          samples_output_path=args.out_samples)

    print("Training finished.")

if __name__ == '__main__':
    #main_toy()
    #exit(0)
    main()
