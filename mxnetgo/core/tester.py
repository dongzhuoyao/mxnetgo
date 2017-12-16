
import mxnet as mx
from mxnetgo.core.module import MutableModule


class Predictor(object):
    def __init__(self, symbol, data_names, label_names,
                 context=mx.cpu(), max_data_shapes=None,
                 provide_data=None, provide_label=None,
                 arg_params=None, aux_params=None):
        self._mod = MutableModule(symbol, data_names, label_names,
                                  context=context, max_data_shapes=max_data_shapes)
        self._mod.bind(provide_data, provide_label, for_training=False)
        self._mod.init_params(arg_params=arg_params, aux_params=aux_params)

    def predict(self, data_batch):
        self._mod.forward(data_batch)
        # [dict(zip(self._mod.output_names, _)) for _ in zip(*self._mod.get_outputs(merge_multi_context=False))]
        return [dict(zip(self._mod.output_names, _)) for _ in zip(*self._mod.get_outputs(merge_multi_context=False))]
