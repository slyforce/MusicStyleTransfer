import mxnet as mx
from VarAutoEncoder.data import Loader, MelodyDataset, Dataset
from music_style_transfer.MIDIUtil.Melody import Melody
from music_style_transfer.MIDIUtil.MelodyWriter import MelodyWriter
from music_style_transfer.MIDIUtil.Note import Note
from . import config
from . import model
from . import utils

import os

def load_model(model_folder: str,
               context: mx.Context,
               checkpoint: int):
    m = model.EncoderDecoder(config.Config.load(model_folder + '/config'))
    utils.load_model_parameters(m, model_folder + '/params.{}'.format(checkpoint), context)
    return m


def setup():
    args = config.get_config()

    loader = Loader(path=args.data,
                    max_sequence_length=args.max_seq_len,
                    slices_per_quarter_note=args.slices_per_quarter_note)

    dataset = MelodyDataset(
        melodies=loader.melodies,
        batch_size=1)

    context = mx.gpu() if args.gpu else mx.cpu()

    m = load_model(args.model_output,
                   context,
                   args.load_checkpoint)

    utils.create_directory_if_not_present(args.out_samples)

    sampler = Sampler(model=m,
                      context=context,
                      visualize_samples=args.visualize_samples,
                      output_path=args.out_samples)
    sampler.sample_from_dataset(dataset, '')


class Sampler:
    def __init__(self,
                 model: model.EncoderDecoder,
                 context: mx.Context,
                 output_path: str,
                 visualize_samples: bool = True):
        self.model = model
        self.context = context
        self.visualize_samples = visualize_samples
        self.melody_writer = MelodyWriter()
        self.output_path = output_path

    def _generate_var_ae_noise(self, batch_size, seq_len, set_to_zero=False):
        shape = (batch_size, seq_len, self.model.config.latent_dimension)
        if set_to_zero:
            return mx.nd.zeros(shape, ctx=self.context)
        else:
            return mx.nd.random_normal(loc=0.,
                                       scale=1.,
                                       shape=shape,
                                       ctx=self.context)

    def sample_from_dataset(self,
                            dataset: Dataset,
                            output_suffix: str):
        for i, batch in enumerate(dataset):
            self.sample_batch(batch, dataset.num_classes(), output_suffix + "melody_id{}_class".format(i))

    def sample_batch(self,
                     batch: mx.io.DataBatch,
                     num_classes: int,
                     output_suffix: str):
        [tokens, articulations, classes] = [x.as_in_context(self.context) for x in batch.data]
        (batch_size, seq_len, _) = tokens.shape

        print("Writing {}".format(os.path.join(self.output_path, '{}_original.mid'.format(output_suffix))))

        self.melody_writer.write_to_file(os.path.join(self.output_path, '{}_original.mid'.format(output_suffix)),
                                         self.construct_melody(tokens[0, :], articulations[0, :]))
        for c in range(0, num_classes):
            tokens_out, articulations_out, _, _ = self.model(tokens,
                                                             articulations,
                                                             classes,
                                                             mx.nd.ones_like(classes) * c,
                                                             self._generate_var_ae_noise(batch_size, seq_len, True))

            print("Writing {}".format(os.path.join(self.output_path, '{}_{}.mid'.format(output_suffix, c))))
            self.melody_writer.write_to_file(os.path.join(self.output_path, '{}_{}.mid'.format(output_suffix, c)),
                                             self.construct_melody(tokens_out[0, :], articulations_out[0, :]))

    def construct_melody(self, tokens, articulations):
        melody = Melody()
        seq_len = tokens.shape[0]

        for i in range(seq_len):
            notes = set()
            active_pitches = [j for j, value in enumerate(tokens[i, :]) if value > 0.]

            for active_pitch in active_pitches:
                notes.add(Note(midi_pitch=active_pitch,
                               articulated=articulations[i, active_pitch] > 0.))

            melody.notes.append(notes)

        if self.visualize_samples:
            utils.visualize_melody(melody)

        return melody

if __name__ == '__main__':
    setup()