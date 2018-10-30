import mxnet as mx
from VarAutoEncoder.data import ToyData, Loader, Dataset, load_dataset
from music_style_transfer.VarAutoEncoder import model
from music_style_transfer.VarAutoEncoder import trainer
from .config import get_config
from .utils import create_directory_if_not_present


def create_toy_model_config(data):
    return model.EncoderDecoderConfig(
        latent_dimension=16,
        encoder_config=model.LSTMConfig(
            n_layers=1,
            hidden_dim=32,
            dropout=0.0),
        decoder_config=model.LSTMConfig(
            n_layers=1,
            hidden_dim=32,
            dropout=0.0),
        feature_dimension=data.num_tokens(),
        input_classes=data.num_classes()
    )


def create_toy_train_config():
    config = trainer.TrainConfig(batch_size=1,
                                 sampling_frequency=500,
                                 checkpoint_frequency=1000,
                                 num_checkpoints_not_improved=-1,
                                 kl_loss=0.0,
                                 optimizer=trainer.OptimizerConfig(
                                     learning_rate=1e-4,
                                     optimizer='adam',
                                     optimizer_params = 'clip_gradient:1.0',
                                 ),
                                 label_smoothing=0.0)
    return config


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
                                 label_smoothing=args.label_smoothing)
    return config


def create_model_config(args, dataset: Dataset):
    return model.EncoderDecoderConfig(
        latent_dimension=args.latent_dim,
        encoder_config=model.LSTMConfig(
            n_layers=args.e_n_layers,
            hidden_dim=args.e_rnn_hidden_dim,
            dropout=args.e_dropout),
        decoder_config=model.LSTMConfig(
            n_layers=args.d_n_layers,
            hidden_dim=args.d_rnn_hidden_dim,
            dropout=args.d_dropout),
        feature_dimension=dataset.num_tokens(),
        input_classes=dataset.num_classes()
    )

def main_toy():
    dataset = ToyData()

    m = model.EncoderDecoder(config=create_toy_model_config(dataset))

    t = trainer.Trainer(config=create_toy_train_config(),
                        context=mx.cpu(),
                        model=m)

    t.fit(dataset=dataset,
          validation_dataset=dataset,
          model_folder='/tmp/out',
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

    train_dataset, valid_dataset = load_dataset(loader.melodies, args.validation_split, args.batch_size)

    create_directory_if_not_present(args.model_output)
    create_directory_if_not_present(args.out_samples)

    config = create_model_config(args, train_dataset)
    config.save(args.model_output + '/config')

    m = model.EncoderDecoder(config=config)

    t = trainer.Trainer(config=create_train_config(args),
                        context=mx.gpu() if args.gpu else mx.cpu(),
                        model=m)

    t.fit(dataset=train_dataset,
          validation_dataset=valid_dataset,
          model_folder=args.model_output,
          epochs=args.epochs,
          samples_output_path=args.out_samples)

    print("Training finished.")

if __name__ == '__main__':
    #main_toy()
    #exit(0)
    main()
