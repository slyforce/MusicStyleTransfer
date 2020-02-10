# -*- coding: utf-8 -*-
import argparse


def str2bool(v):
    return v.lower() in ('true', '1')


arg_lists = []
parser = argparse.ArgumentParser()


def add_argument_group(name):
    arg = parser.add_argument_group(name)
    arg_lists.append(arg)
    return arg


# Network
net_arg = add_argument_group('Network')
net_arg.add_argument('--e-n-layers', type=int, default=1)
net_arg.add_argument('--e-rnn-hidden-dim', type=int, default=128)
net_arg.add_argument('--e-emb-hidden-dim', type=int, default=64)
net_arg.add_argument('--e-dropout', type=float, default=0.0)
net_arg.add_argument('--e-num-heads', type=int, default=8)

net_arg.add_argument('--latent-dim', type=int, default=64)

net_arg.add_argument('--d-n-layers', type=int, default=1)
net_arg.add_argument('--d-rnn-hidden-dim', type=int, default=128)
net_arg.add_argument('--d-dropout', type=float, default=0.0)

# Data
data_arg = add_argument_group('Data')
data_arg.add_argument('--batch-size', type=int, default=1)
data_arg.add_argument('--max-seq-len', type=int, default=64)
data_arg.add_argument('--slices-per-quarter-note', type=float, default=4)
data_arg.add_argument('--data', type=str, default='data')
data_arg.add_argument('--validation-data', type=str, default=None)
data_arg.add_argument('--minimum-pattern-length', type=int, default=16)
data_arg.add_argument('--pattern-identifier', type=str, choices=['recurring', ''], default='')


# Training / test parameters
train_arg = add_argument_group('Training')
train_arg.add_argument('--epochs', type=int, default=5000)
train_arg.add_argument('--learning-rate', type=float, default=3e-4)
train_arg.add_argument('--optimizer', type=str, default='adam')
train_arg.add_argument('--optimizer-params', type=str, default='')
train_arg.add_argument('--validation-split', type=float, default=0.1)
train_arg.add_argument('--kl-loss', type=float, default=1.0)
train_arg.add_argument('--label-smoothing', type=float, default=0.0)
train_arg.add_argument('--negative-label-downscaling', action='store_true')
train_arg.add_argument('--beam-size', type=int, default=5)
train_arg.add_argument('--sampling-type', choices=["beam-search", "sampling"], default="sampling")


# Misc
misc_arg = add_argument_group('Misc')
misc_arg.add_argument('--load-checkpoint', type=int, default=1)
misc_arg.add_argument('--checkpoint-frequency', type=int, default=5000)
misc_arg.add_argument('--sampling-frequency', type=int, default=1000)
misc_arg.add_argument('--num-checkpoints-not-improved', type=int, default=10)
misc_arg.add_argument('--out-samples', '-o', type=str, default=None)
misc_arg.add_argument('--model-output', '-m', type=str, default='models')
misc_arg.add_argument('--checkpoint', '-c', type=int, default=-1)
misc_arg.add_argument('--gpu', action='store_true')
misc_arg.add_argument('--toy', action='store_true')
misc_arg.add_argument('--visualize-samples', action='store_true')
misc_arg.add_argument('--verbose',  action='store_true')


def get_config():
    config, unparsed = parser.parse_known_args()
    return config

import copy
import inspect
import yaml

class TaggedYamlObjectMetaclass(yaml.YAMLObjectMetaclass):
    def __init__(cls, name, bases, kwds):
        cls.yaml_tag = "!" + name
        new_kwds = {}
        new_kwds.update(kwds)
        new_kwds['yaml_tag'] = "!" + name
        super().__init__(name, bases, new_kwds)


class Config(yaml.YAMLObject, metaclass=TaggedYamlObjectMetaclass):
    """
    Base configuration object that supports freezing of members and YAML (de-)serialization.
    Actual Configuration should subclass this object.
    """
    def __init__(self):
        self.__add_frozen()

    def __setattr__(self, key, value):
        if hasattr(self, '_frozen') and getattr(self, '_frozen'):
            raise AttributeError("Cannot set '%s' in frozen config" % key)
        if value == self:
            raise AttributeError("Cannot set self as attribute")
        object.__setattr__(self, key, value)

    def __setstate__(self, state):
        """Pickle protocol implementation."""
        # We first take the serialized state:
        self.__dict__.update(state)
        # Then we take the constructors default values for missing arguments in order to stay backwards compatible
        # This way we can add parameters to Config objects and still load old models.
        init_signature = inspect.signature(self.__init__)
        for param_name, param in init_signature.parameters.items():
            if param.default is not param.empty:
                if not hasattr(self, param_name):
                    object.__setattr__(self, param_name, param.default)

    def freeze(self):
        """
        Freezes this Config object, disallowing modification or addition of any parameters.
        """
        if getattr(self, '_frozen'):
            return
        object.__setattr__(self, "_frozen", True)
        for k, v in self.__dict__.items():
            if isinstance(v, Config) and k != "self":
                v.freeze()  # pylint: disable= no-member


    def __repr__(self):
        return "Config[%s]" % ", ".join("%s=%s" % (str(k), str(v)) for k, v in sorted(self.__dict__.items()))


    def __eq__(self, other):
        if type(other) is not type(self):
            return False
        for k, v in self.__dict__.items():
            if k != "self":
                if k not in other.__dict__:
                    return False
                if self.__dict__[k] != other.__dict__[k]:
                    return False
        return True


    def __del_frozen(self):
        """
        Removes _frozen attribute from this instance and all its child configurations.
        """
        self.__delattr__('_frozen')
        for attr, val in self.__dict__.items():
            if isinstance(val, Config) and hasattr(val, '_frozen'):
                val.__del_frozen()  # pylint: disable= no-member


    def __add_frozen(self):
        """
        Adds _frozen attribute to this instance and all its child configurations.
        """
        setattr(self, "_frozen", False)
        for attr, val in self.__dict__.items():
            if isinstance(val, Config):
                val.__add_frozen()  # pylint: disable= no-member



    def save(self, fname: str):
        """
        Saves this Config (without the frozen state) to a file called fname.

        :param fname: Name of file to store this Config in.
        """
        obj = copy.deepcopy(self)
        obj.__del_frozen()
        with open(fname, 'w') as out:
            yaml.dump(obj, out, default_flow_style=False)


    @staticmethod
    def load(fname: str) -> 'Config':
        """
        Returns a Config object loaded from a file. The loaded object is not frozen.

        :param fname: Name of file to load the Config from.
        :return: Configuration.
        """
        with open(fname) as inp:
            obj = yaml.load(inp)
            obj.__add_frozen()
            return obj


    def copy(self, **kwargs):
        """
        Create a copy of the config object, optionally modifying some of the attributes.
        For example `nn_config.copy(num_hidden=512)` will create a copy of `nn_config` where the attribute `num_hidden`
        will be set to the new value of num_hidden.

        :param kwargs:
        :return: A deep copy of the config object.
        """
        copy_obj = copy.deepcopy(self)
        for name, value in kwargs.items():
            object.__setattr__(copy_obj, name, value)
        return copy_obj

    def set_attrs(self, attrs):
        """ Adds all attrs to self, used in constructor e.gl:
        self.set_attrs(locals()) """
        for k, v in attrs.items():
            if k == 'self' and self == v:
                # Ignore self
                pass
            elif hasattr(self, k):
                print(
                    'Not automatically over writing setting '
                    '%s, %s. %s is already defined for Object %s' %
                    (k, str(v), k, self))
            else:
                setattr(self, k, v)

    def output_to_stream(self, stream):
        yaml.dump(self, stream)