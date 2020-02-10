import mxnet as mx
from VarAutoEncoder.data import ToyData, Loader, Dataset, load_dataset
from VarAutoEncoder.utils import log_config, log_model_variables
from music_style_transfer.VarAutoEncoder import model
from .transformer import TransformerConfig
from music_style_transfer.VarAutoEncoder import trainer
from .config import get_config
from .utils import create_directory_if_not_present
from .sampler import get_sampler

import os


def create_toy_model_config(data):
    return model.ModelConfig(
        encoder_config=model.EncoderConfig(
            transformer_config=TransformerConfig(
                model_size=32,
                dropout=0.0,
                num_layers=1,
                vocab_size=data.num_tokens(),
                num_heads=2),
            latent_dim=16,
            num_classes=data.num_classes(),
            input_dim=data.num_tokens(),
        ),
        decoder_config=model.DecoderConfig(
            transformer_config=TransformerConfig(
                model_size=32,
                dropout=0.0,
                num_layers=1,
                vocab_size=data.num_tokens(),
                num_heads=2),
            latent_dim=16,
            num_classes=data.num_classes(),
            output_dim=data.num_tokens()
        )
    )


def create_toy_train_config():
    config = trainer.TrainConfig(batch_size=1,
                                 sampling_frequency=500,
                                 checkpoint_frequency=1000,
                                 num_checkpoints_not_improved=-1,
                                 kl_loss=1.0,
                                 optimizer=trainer.OptimizerConfig(
                                     learning_rate=1e-3,
                                     optimizer='adam',
                                     optimizer_params='clip_gradient:1.0',
                                 ),
                                 label_smoothing=0.0,
                                 negative_label_downscaling=True,
                                 verbose=False)
    return config


def main_toy():
    dataset = ToyData()

    config = create_toy_model_config(dataset)
    m = model.Model(config=config)
    model_folder = "/tmp/music-style-transfer/toy/model"

    create_directory_if_not_present(model_folder)
    config.save(os.path.join(model_folder, 'config'))

    t = trainer.Trainer(config=create_toy_train_config(),
                        context=mx.cpu(),
                        model=m,
                        sampler=None)

    t.fit(dataset=dataset,
          validation_dataset=dataset,
          model_folder=model_folder,
          epochs=20000)


def create_train_config(args):
    config = trainer.TrainConfig(batch_size=args.batch_size,
                                 sampling_frequency=args.sampling_frequency,
                                 checkpoint_frequency=args.checkpoint_frequency,
                                 num_checkpoints_not_improved=args.num_checkpoints_not_improved,
                                 kl_loss=args.kl_loss,
                                 optimizer=trainer.OptimizerConfig(
                                     learning_rate=args.learning_rate,
                                     optimizer=args.optimizer,
                                     optimizer_params=args.optimizer_params,
                                 ),
                                 label_smoothing=args.label_smoothing,
                                 negative_label_downscaling=args.negative_label_downscaling,
                                 verbose=args.verbose)
    return config


def create_model_config(args, dataset: Dataset):
    return model.ModelConfig(
        encoder_config=model.EncoderConfig(
            transformer_config=TransformerConfig(
                model_size=args.e_rnn_hidden_dim,
                dropout=args.e_dropout,
                num_layers=args.e_n_layers,
                vocab_size=dataset.num_tokens(),
                num_heads=args.e_num_heads),
            latent_dim=args.latent_dim,
            num_classes=dataset.num_classes(),
            input_dim=dataset.num_tokens(),
        ),
        decoder_config=model.DecoderConfig(
            lstm_config=model.LSTMConfig(
                n_layers=args.d_n_layers,
                hidden_dim=args.d_rnn_hidden_dim,
                dropout=args.d_dropout),
            latent_dim=args.latent_dim,
            num_classes=dataset.num_classes(),
            output_dim=dataset.num_tokens()
        )
    )



def main():
    args = get_config()
    context = mx.gpu() if args.gpu else mx.cpu()

    if args.toy:
        main_toy()
        exit(0)

    loader = Loader(path=args.data,
                    max_sequence_length=args.max_seq_len,
                    slices_per_quarter_note=args.slices_per_quarter_note)

    if args.validation_data is not None:
        val_loader = Loader(path=args.validation_data,
                            max_sequence_length=args.max_seq_len,
                            slices_per_quarter_note=args.slices_per_quarter_note)
    else:
        val_loader = None

    train_dataset, valid_dataset = load_dataset(loader,
                                                args.batch_size,
                                                args.validation_split,
                                                val_loader)

    create_directory_if_not_present(args.model_output)
    create_directory_if_not_present(args.out_samples)

    config = create_model_config(args, train_dataset)
    config.save(args.model_output + '/config')
    log_config(config)

    m = model.Model(config=config)
    log_model_variables(m)

    sampler = get_sampler('sampling',
                          args.model_output,
                          context,
                          None,
                          args)

    t = trainer.Trainer(config=create_train_config(args),
                        context=context,
                        model=m,
                        sampler=sampler)

    t.fit(dataset=train_dataset,
          validation_dataset=valid_dataset,
          model_folder=args.model_output,
          epochs=args.epochs)

    print("Training finished.")


if __name__ == '__main__':
    main()
