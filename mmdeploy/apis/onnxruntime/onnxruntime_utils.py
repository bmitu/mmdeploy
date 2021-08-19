import os.path as osp

import numpy as np
import onnxruntime as ort
import torch


class ORTWrapper(torch.nn.Module):
    """ONNXRuntime Wrapper.

    Arguments:
        onnx_file (str): Input onnx model file
        device_id (int): The device id to put model
    """

    def __init__(self, onnx_file: str, device_id: int):
        super(ORTWrapper, self).__init__()
        # get the custom op path
        from mmdeploy.apis.onnxruntime import get_ops_path
        ort_custom_op_path = get_ops_path()
        session_options = ort.SessionOptions()
        # register custom op for onnxruntime
        if osp.exists(ort_custom_op_path):
            session_options.register_custom_ops_library(ort_custom_op_path)
        sess = ort.InferenceSession(onnx_file, session_options)

        providers = ['CPUExecutionProvider']
        options = [{}]
        is_cuda_available = ort.get_device() == 'GPU'
        if is_cuda_available:
            providers.insert(0, 'CUDAExecutionProvider')
            options.insert(0, {'device_id': device_id})
        sess.set_providers(providers, options)

        self.sess = sess
        self.io_binding = sess.io_binding()
        self.output_names = [_.name for _ in sess.get_outputs()]
        self.device_id = device_id
        self.is_cuda_available = is_cuda_available
        self.device_type = 'cuda' if is_cuda_available else 'cpu'

    def forward(self, inputs):
        """
        Arguments:
            inputs (tensor): the input tensor

        Return:
            dict: dict of output name-tensors pair
        """
        # set io binding for inputs/outputs
        if not self.is_cuda_available:
            inputs = inputs.cpu()
        self.io_binding.bind_input(
            name='input',
            device_type=self.device_type,
            device_id=self.device_id,
            element_type=np.float32,
            shape=inputs.shape,
            buffer_ptr=inputs.data_ptr())

        for name in self.output_names:
            self.io_binding.bind_output(name)
        # run session to get outputs
        self.sess.run_with_iobinding(self.io_binding)
        outputs = self.io_binding.copy_outputs_to_cpu()

        return outputs
