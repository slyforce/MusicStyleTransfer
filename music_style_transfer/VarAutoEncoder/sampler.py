import mxnet as mx
from VarAutoEncoder.data import Loader, MelodyDataset
from music_style_transfer.MIDIUtil.Melody import Melody
from music_style_transfer.MIDIUtil.MelodyWriter import MelodyWriter
from music_style_transfer.MIDIUtil.Note import Note
from . import config
from . import model
from . import utils


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

    sampler = Sampler(m, context)
    sampler.sample_from_dataset(dataset, args.out_samples)

class Sampler:
    def __init__(self,
                 model,
                 context):
        self.model = model
        self.context = context

        self.melody_writer = MelodyWriter()

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
                            dataset,
                            output_path):
        for batch in dataset:
            self.sample_batch(batch, dataset.num_classes(), output_path)

    def sample_batch(self, batch, num_classes, output_path):
        [tokens, articulations, classes] = [x.as_in_context(self.context) for x in batch.data]
        (batch_size, seq_len, _) = tokens.shape
        self.melody_writer.write_to_file(output_path + '_original.mid',
                                         self.construct_melody(tokens[0, :], articulations[0, :]))
        for c in range(0, num_classes):
            tokens_out, articulations_out, _, _ = self.model(tokens,
                                                             articulations,
                                                             classes,
                                                             mx.nd.ones_like(classes) * c,
                                                             self._generate_var_ae_noise(batch_size, seq_len, True))

            self.melody_writer.write_to_file(output_path + '_{}.mid'.format(c),
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

        utils.visualize_melody(melody)
        return melody

if __name__ == '__main__':
    setup()