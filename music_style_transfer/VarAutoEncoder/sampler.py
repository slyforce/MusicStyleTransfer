import mxnet as mx
from VarAutoEncoder.data import Loader, MelodyDataset, Dataset
from music_style_transfer.MIDIUtil.Melody import Melody
from music_style_transfer.MIDIUtil.defaults import *

from . import config
from . import model
from . import utils


import os


def load_inference_model(model_folder: str,
                         context: mx.Context,
                         checkpoint: int):

    # 1. load the configuration
    c = config.Config.load(os.path.join(model_folder, 'config')) # type: model.ModelConfig
    encoder = model.Encoder(c.encoder_config)
    decoder = model.InferenceDecoder(c.decoder_config)

    # 2. load the parameters
    params_fname = os.path.join(model_folder, 'params.{}'.format(checkpoint))
    utils.load_model_parameters(encoder, params_fname, context)
    utils.load_model_parameters(decoder, params_fname, context)
    return encoder, decoder


class SamplerBase:
    def __init__(self,
                 encoder: model.Encoder,
                 decoder: model.InferenceDecoder,
                 context: mx.Context):
        self.encoder = encoder
        self.decoder = decoder
        self.context = context

    def _check_input(self, batch: mx.io.DataBatch):
        at_least_one_error = False

        if len(batch.data) != 3:
            at_least_one_error = True

        if len(batch.label) != 1:
            at_least_one_error = True

        if at_least_one_error:
            raise ValueError("Input to sampler is wrong: {}".format(batch))

    def read_batch(self, batch: mx.io.DataBatch):
        data = [x.as_in_context(self.context) for x in batch.data]
        labels = [x.as_in_context(self.context) for x in batch.label]
        return data, labels

    def sample(self, data_batch: mx.io.DataBatch):
        self._check_input(data_batch)
        raise NotImplemented

    def compute_initial_decoder_state(self, tokens, seq_lens, classes):
        means, vars = self.encoder(tokens, seq_lens, classes)
        # todo: implement more variants of this
        latent_vector = means

        return self.decoder.get_initial_state(mx.nd, classes, latent_vector)


class Sampling(SamplerBase):
    def __init__(self,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)

    def sample(self, data_batch: mx.io.DataBatch):
        self._check_input(data_batch)
        [tokens, seq_lens, classes], _ = self.read_batch(data_batch)

        I_max = mx.nd.max(seq_lens) * 2 # todo: do not hard-code this
        beam_size = tokens.shape[0]

        sequences = mx.nd.zeros((beam_size, I_max))
        sequences[:, 0] = SOS_ID
        scores = mx.nd.zeros((beam_size,))

        previous_decoder_state = self.compute_initial_decoder_state(tokens, seq_lens, classes)

        for i in range(1, I_max):

            prev_tokens = sequences[:,i-1]
            probs, next_states = self.decoder(prev_tokens, previous_decoder_state, classes, i)

            next_outputs = mx.nd.random.multinomial(probs)

            sequences[:, i] = next_outputs
            scores += mx.nd.pick(probs, next_outputs)

            previous_decoder_state = next_states

        return sequences


class BeamSearchSampler(SamplerBase):
    def __init__(self,
                 beam_size: int,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.beam_size = beam_size

    def sample(self, data_batch: mx.io.DataBatch):
        raise NotImplemented






def setup():
    args = config.get_config()

    loader = Loader(path=args.data,
                    max_sequence_length=args.max_seq_len,
                    slices_per_quarter_note=args.slices_per_quarter_note)

    dataset = MelodyDataset(
        melodies=loader.melodies,
        batch_size=1)

    context = mx.gpu() if args.gpu else mx.cpu()

    m = load_inference_model(args.model_output,
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
                 model: model.Decoder,
                 context: mx.Context,
                 output_path: str,
                 visualize_samples: bool = True):
        self.model = model
        self.context = context
        self.visualize_samples = visualize_samples
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