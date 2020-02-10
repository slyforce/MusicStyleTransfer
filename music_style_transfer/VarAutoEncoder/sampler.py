import mxnet as mx
from VarAutoEncoder.data import Loader, MelodyDataset, Dataset
from music_style_transfer.MIDIUtil.midi_io import MelodyWriter
from music_style_transfer.MIDIUtil.Melody import Melody, get_melody_from_ids
from music_style_transfer.MIDIUtil.defaults import *

from . import config
from . import model
from . import utils

from typing import Optional

import re
import os


def load_inference_model(model_folder: str,
                         context: mx.Context,
                         checkpoint: int):

    # 1. load the configuration
    c = config.Config.load(os.path.join(model_folder, 'config')) # type: model.ModelConfig
    utils.log_config(c)

    m = model.Model(c)
    utils.log_model_variables(m)

    if checkpoint is None:
        return m

    # 2. load the parameters
    if checkpoint == -1:
        # pick latest checkpoint
        checkpoint = utils.get_latest_checkpoint_index(model_folder)

    params_fname = os.path.join(model_folder, 'params.{}'.format(checkpoint))
    utils.load_model_parameters(m, params_fname, context)
    return m


def get_sampler(type: str,
                model_folder: str,
                context: mx.Context,
                checkpoint: Optional[int],
                args):
    if type == 'sampling':
        return Sampling(model_folder, context, checkpoint,
                        verbose=args.verbose)
    elif type == 'beam-search':
        return BeamSearchSampler(model_folder, context, checkpoint,
                                 beam_size=args.beam_size, verbose=args.verbose)
    else:
        raise ValueError("Sampler {} is not implemented".format(type))


class SamplerBase:
    def __init__(self,
                 model_folder: str,
                 context: mx.Context,
                 checkpoint: int,
                 verbose: bool = False):
        self.model = load_inference_model(model_folder, context, checkpoint)
        self.encoder, self.decoder = self.model.encoder, self.model.decoder
        self.model_folder = model_folder
        self.context = context
        self.verbose = verbose

    def reload_checkpoint(self, checkpoint: int):
        self.model = load_inference_model(self.model_folder, self.context, checkpoint)
        self.encoder, self.decoder = self.model.encoder, self.model.decoder

    def update_parameters(self, model: model.Model):
        self.model = model
        self.encoder = self.model.encoder
        self.decoder = self.model.decoder

    def process_dataset(self,
                        dataset: Dataset,
                        output_suffix: str):
        utils.create_directory_if_not_present(output_suffix)

        print("Starting to decode dataset")
        writer = MelodyWriter()
        current_sample_idx = 0
        for i, batch in enumerate(dataset):
            print("Processing batch {}".format(i))

            # write original versions of the data
            for i, sequence in enumerate(mx.nd.split(batch.data[0], axis=0, num_outputs=batch.data[0].shape[0])):
                writer.write_to_file(os.path.join(output_suffix, "out-{}.original.mid".format(current_sample_idx + i)),
                                     get_melody_from_ids(sequence.squeeze().asnumpy()))

            for class_idx in range(dataset.num_classes()):
                # generate a sample for each output class
                batch.data[2] = mx.nd.full(shape=batch.data[2].shape, val=class_idx)

                # generate samples
                sequences = self.sample(batch)

                # now output each sequence in batch
                for i, sequence in enumerate(mx.nd.split(sequences, axis=0, num_outputs=sequences.shape[0])):
                    writer.write_to_file(os.path.join(output_suffix, "out-{}.class-{}.mid".format(current_sample_idx + i,
                                                                                                  class_idx)),
                                         get_melody_from_ids(sequence.squeeze().asnumpy()))

            # increment by the batch size
            current_sample_idx += batch.data[0].shape[0]

        print("Done with dataset decoding")

    def process_batch(self,
                      batch: mx.io.DataBatch,
                      output_suffix: str,
                      num_classes: int):

        utils.create_directory_if_not_present(output_suffix)
        writer = MelodyWriter()

        for i, sequence in enumerate(mx.nd.split(batch.data[0], axis=0, squeeze_axis=True, num_outputs=batch.data[0].shape[0])):
            writer.write_to_file(os.path.join(output_suffix, "out-{}.original.mid".format(i)),
                                 get_melody_from_ids(sequence.asnumpy()))

        for class_idx in range(num_classes):

            # generate a sample for each output class
            batch.data[2] = mx.nd.full(shape=batch.data[2].shape, val=class_idx)

            # generate samples
            sequences = self.sample(batch)

            # now output each sequence in batch
            for i, sequence in enumerate(mx.nd.split(sequences, axis=0, squeeze_axis=True, num_outputs=sequences.shape[0])):
                writer.write_to_file(os.path.join(output_suffix, "out-{}.class-{}.mid".format(i,
                                                                                              class_idx)),
                                     get_melody_from_ids(sequence.asnumpy()))

    def read_batch(self, batch: mx.io.DataBatch):
        data = [x.as_in_context(self.context) for x in batch.data]
        labels = [x.as_in_context(self.context) for x in batch.label]
        return data, labels

    def sample(self, data_batch: mx.io.DataBatch):
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
        [tokens, seq_lens, classes], _ = self.read_batch(data_batch)

        I_max = tokens.shape[1] * 2 # todo: do not hard-code this
        beam_size = tokens.shape[0]

        sequences = mx.nd.full((beam_size, I_max), val=PAD_ID)
        sequences[:, 0] = SOS_ID
        scores = mx.nd.zeros((beam_size,))
        previous_decoder_state = self.compute_initial_decoder_state(tokens, seq_lens, classes)

        if self.verbose:
            print("Inputs to sampling: ")
            print("Tokens: {}, {}".format(tokens.shape, tokens))
            print("seq_lens: {}, {}".format(seq_lens.shape, seq_lens))
            print("classes: {}, {}".format(classes.shape, classes))

        for i in range(1, I_max):

            prev_tokens = sequences[:, i-1]
            probs, next_states = self.decoder.forward_inference(prev_tokens, previous_decoder_state, classes, i)

            next_outputs = mx.nd.random.multinomial(probs)

            sequences[:, i] = next_outputs
            scores += -mx.nd.log(mx.nd.pick(probs, next_outputs))

            previous_decoder_state = next_states

            if mx.nd.sum(mx.nd.broadcast_logical_or(sequences[:,i] == SOS_ID, sequences[:, i] == PAD_ID)).asscalar() == beam_size:
                break

        return sequences


class BeamSearchSampler(SamplerBase):
    def __init__(self,
                 *args,
                 beam_size: int,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.beam_size = beam_size
        self.max_length_factor = 2.

    def sample(self, data_batch: mx.io.DataBatch):
        [tokens, seq_lens, classes], _ = self.read_batch(data_batch)

        I_max = int(tokens.shape[1] * self.max_length_factor)
        batch_size = tokens.shape[0]

        sequences = mx.nd.full((batch_size * self.beam_size , I_max), val=PAD_ID)
        sequences[:, 0] = SOS_ID
        scores = mx.nd.zeros((batch_size * self.beam_size,))

        # offset for hypothesis indices in batch decoding
        offset = mx.nd.repeat(mx.nd.arange(0, batch_size * self.beam_size, self.beam_size,
                                           dtype='int32', ctx=self.context), self.beam_size)

        previous_decoder_states = [mx.nd.repeat(state,
                                                repeats=self.beam_size,
                                                axis=1) for state in self.compute_initial_decoder_state(tokens, seq_lens, classes)]

        for i in range(1, I_max):

            prev_tokens = sequences[:, i-1]
            probs, next_states = self.decoder.forward_inference(prev_tokens, previous_decoder_states, classes, i)
            expansion_scores = -mx.nd.log(probs)

            # update scores only for un-finished hypotheses
            expansion_scores = mx.nd.where(mx.nd.broadcast_logical_or(sequences[:,i-1] == EOS_ID, sequences[:, i-1] == PAD_ID),
                                           mx.nd.zeros_like(expansion_scores) * 0,
                                           expansion_scores)
            expansion_scores = mx.nd.broadcast_add(scores.expand_dims(axis=1), expansion_scores)

            folded_scores = mx.nd.reshape(expansion_scores, shape=(batch_size, -1))

            top_k_scores, top_k_indices = mx.nd.topk(folded_scores, axis=1, k=self.beam_size, ret_typ='both', is_ascend=True)
            #print(top_k_scores)

            best_hyp_indices, best_word_indices = mx.nd.split(mx.nd.unravel_index(mx.nd.cast(top_k_indices.reshape((-1,)), 'int32'),
                                                                                  shape=(self.beam_size, NUM_EVENTS)),
                                                              num_outputs=2, squeeze_axis=True, axis=0)
            best_hyp_indices += offset
            #print(best_hyp_indices, best_word_indices, top_k_indices)

            sequences = sequences.take(best_hyp_indices)
            sequences[:, i] = best_word_indices
            scores = scores.take(best_hyp_indices) + mx.nd.reshape(top_k_scores, (-1,))
            for i in range(len(previous_decoder_states)):
                previous_decoder_states[i] = previous_decoder_states[i].take(best_hyp_indices, axis=1)

            for batch_idx in range(batch_size):
                for beam_idx in range(self.beam_size):
                    f = batch_idx * self.beam_size + beam_idx
                    #print("{}-{} # {} # {}".format(batch_idx, beam_idx, scores[f].asscalar(), list(sequences[f,:].asnumpy())))

            if mx.nd.sum(mx.nd.broadcast_logical_or(sequences[:,i] == EOS_ID, sequences[:, i] == PAD_ID)).asscalar() == batch_size:
                break

        # take every k-th hypothesis
        return sequences

from .data import ToyData

def sample_toy(args):
    sampler = get_sampler("sampling",
                          args.model_output,
                          mx.cpu(),
                          args.checkpoint,
                          args)

    dataset = ToyData()

    sampler.process_dataset(dataset, args.out_samples)


def main():
    args = config.get_config()
    context = mx.gpu() if args.gpu else mx.cpu()

    if args.toy:
        sample_toy(args)
        exit(0)

    loader = Loader(path=args.data,
                    max_sequence_length=args.max_seq_len,
                    slices_per_quarter_note=args.slices_per_quarter_note)
    dataset = MelodyDataset(args.batch_size, loader.max_sequence_length, loader.melodies)

    sampler = get_sampler(args.sampling_type,
                          args.model_output,
                          context,
                          args.checkpoint,
                          args)

    sampler.process_dataset(dataset, args.out_samples)


if __name__ == '__main__':
    main()