import mxnet as mx

from music_style_transfer.MIDIUtil.MelodyWriter import MelodyWriter

from music_style_transfer.GAN.data import Loader, MelodyDataset

from . import config
from . import model
from . import utils

import os

def load_models(model_folder: str,
                context: mx.Context,
                checkpoint: int):
    assert os.path.exists(model_folder)
    assert os.path.exists(model_folder + '/decoder/')
    assert os.path.exists(model_folder+ '/encoder/')

    d_config = config.Config.load(model_folder + '/decoder/config')
    e_config = config.Config.load(model_folder + '/encoder/config')

    decoder = model.Decoder(config=d_config)
    encoder = model.Encoder(config=e_config)

    utils.load_model_parameters(decoder, model_folder + '/decoder/params.{}'.format(checkpoint), context)
    utils.load_model_parameters(encoder, model_folder + '/encoder/params.{}'.format(checkpoint), context)
    return decoder, encoder


def setup():
    args = config.get_config()

    loader = Loader(path=args.data,
                    max_sequence_length=args.max_seq_len,
                    slices_per_quarter_note=args.slices_per_quarter_note)

    dataset = MelodyDataset(
        melodies=loader.melodies,
        batch_size=1)

    context = mx.gpu() if args.gpu else mx.cpu()

    decoder, encoder = load_models(args.model_output,
                                   context,
                                   args.load_checkpoint)

    sampler = Sampler(decoder, encoder, context)
    sampler.sample_from_dataset(dataset, args.out_samples)

class Sampler:
    def __init__(self,
                 decoder,
                 encoder,
                 context):
        self.decoder = decoder
        self.encoder = encoder
        self.context = context

        self.melody_writer = MelodyWriter()

    def sample_from_dataset(self,
                            dataset,
                            output_path):
        for batch in dataset:
            [tokens, classes] = batch.data[0].as_in_context(self.context), batch.data[1].as_in_context(self.context)

            z_mean, z_var = self.encoder(tokens, classes)
            pitches_per_class = []
            for c in range(0, dataset.num_classes()):
                sampled_eps = mx.nd.random_normal(0, scale=1.0, shape=z_mean.shape, ctx=self.context)
                z = z_mean + z_var * sampled_eps
                generated = self.decoder(z, mx.nd.array([c], ctx=self.context))

                # take the maximum over all classes
                pitches_per_class += [mx.nd.argmax(generated, axis=2).asnumpy().ravel()]

            original_melody = utils.construct_melody_from_integers([int(x) for x in tokens.asnumpy().ravel()])
            print("\nOriginal melody: {}".format(original_melody.notes))
            self.melody_writer.write_to_file(output_path + '_original.mid', original_melody)

            for class_index, pitches in enumerate(pitches_per_class):
                melody = utils.construct_melody_from_integers([int(x) for x in pitches])

                print("Melody for class {}: {}".format(class_index, melody.notes))
                self.melody_writer.write_to_file(output_path + '_{}.mid'.format(class_index), melody)


if __name__ == '__main__':
    setup()