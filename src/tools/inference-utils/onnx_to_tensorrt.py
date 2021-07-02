# https://github.com/jkjung-avt/tensorrt_demos
# This file is in the above repository and references code of it
# Run this file in the cloned repository above to get no errors
# My laptop does not support tensorrt, run directly on Nano

#MIT License
#
#Copyright (c) 2019 JK Jung
#
#Permission is hereby granted, free of charge, to any person obtaining a copy
#of this software and associated documentation files (the "Software"), to deal
#in the Software without restriction, including without limitation the rights
#to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#copies of the Software, and to permit persons to whom the Software is
#furnished to do so, subject to the following conditions:
#
#The above copyright notice and this permission notice shall be included in all
#copies or substantial portions of the Software.
#
#THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#SOFTWARE.

from __future__ import print_function

import os
import argparse

import tensorrt as trt

from plugins import get_input_wh, add_yolo_plugins


MAX_BATCH_SIZE = 1


def load_onnx(model_name):
    """Read the ONNX file."""
    onnx_path = '%s.onnx' % model_name
    if not os.path.isfile(onnx_path):
        print('ERROR: file (%s) not found!  You might want to run yolo_to_onnx.py first to generate it.' % onnx_path)
        return None
    else:
        with open(onnx_path, 'rb') as f:
            return f.read()


def set_net_batch(network, batch_size):
    """Set network input batch size.

    The ONNX file might have been generated with a different batch size,
    say, 64.
    """
    if trt.__version__[0] >= '7':
        shape = list(network.get_input(0).shape)
        shape[0] = batch_size
        network.get_input(0).shape = shape
    return network


def build_engine(model_name, category_num, do_int8, dla_core, verbose=False):
    """Build a TensorRT engine from ONNX using the older API."""
    net_w, net_h = get_input_wh(model_name)

    print('Loading the ONNX file...')
    onnx_data = load_onnx(model_name)
    if onnx_data is None:
        return None

    TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE) if verbose else trt.Logger()
    EXPLICIT_BATCH = [] if trt.__version__[0] < '7' else \
        [1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)]
    with trt.Builder(TRT_LOGGER) as builder, builder.create_network(*EXPLICIT_BATCH) as network, trt.OnnxParser(network, TRT_LOGGER) as parser:
        if do_int8 and not builder.platform_has_fast_int8:
            raise RuntimeError('INT8 not supported on this platform')
        if not parser.parse(onnx_data):
            print('ERROR: Failed to parse the ONNX file.')
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return None
        network = set_net_batch(network, MAX_BATCH_SIZE)

        print('Adding yolo_layer plugins...')
        network = add_yolo_plugins(
            network, model_name, category_num, TRT_LOGGER)

        print('Building an engine.  This would take a while...')
        print('(Use "--verbose" or "-v" to enable verbose logging.)')
        if trt.__version__[0] < '7':  # older API: build_cuda_engine()
            if dla_core >= 0:
                raise RuntimeError('DLA core not supported by old API')
            builder.max_batch_size = MAX_BATCH_SIZE
            builder.max_workspace_size = 1 << 30
            builder.fp16_mode = True  # alternative: builder.platform_has_fast_fp16
            if do_int8:
                from calibrator import YOLOEntropyCalibrator
                builder.int8_mode = True
                builder.int8_calibrator = YOLOEntropyCalibrator(
                    'calib_images', (net_h, net_w), 'calib_%s.bin' % model_name)
            engine = builder.build_cuda_engine(network)
        else:  # new API: build_engine() with builder config
            builder.max_batch_size = MAX_BATCH_SIZE
            config = builder.create_builder_config()
            config.max_workspace_size = 1 << 30
            config.set_flag(trt.BuilderFlag.GPU_FALLBACK)
            config.set_flag(trt.BuilderFlag.FP16)
            profile = builder.create_optimization_profile()
            profile.set_shape(
                '000_net',                          # input tensor name
                (MAX_BATCH_SIZE, 3, net_h, net_w),  # min shape
                (MAX_BATCH_SIZE, 3, net_h, net_w),  # opt shape
                (MAX_BATCH_SIZE, 3, net_h, net_w))  # max shape
            config.add_optimization_profile(profile)
            if do_int8:
                from calibrator import YOLOEntropyCalibrator
                config.set_flag(trt.BuilderFlag.INT8)
                config.int8_calibrator = YOLOEntropyCalibrator(
                    'calib_images', (net_h, net_w),
                    'calib_%s.bin' % model_name)
                config.set_calibration_profile(profile)
            if dla_core >= 0:
                config.default_device_type = trt.DeviceType.DLA
                config.DLA_core = dla_core
                config.set_flag(trt.BuilderFlag.STRICT_TYPES)
                print('Using DLA core %d.' % dla_core)
            engine = builder.build_engine(network, config)

        if engine is not None:
            print('Completed creating engine.')
        return engine


def main():
    """Create a TensorRT engine for ONNX-based YOLO."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-v', '--verbose', action='store_true',
        help='enable verbose output (for debugging)')
    parser.add_argument(
        '-c', '--category_num', type=int, default=80,
        help='number of object categories [80]')
    parser.add_argument(
        '-m', '--model', type=str, required=True,
        help=('[yolov3|yolov3-tiny|yolov3-spp|yolov4|yolov4-tiny]-'
              '[{dimension}], where dimension could be a single '
              'number (e.g. 288, 416, 608) or WxH (e.g. 416x256)'))
    parser.add_argument(
        '--int8', action='store_true',
        help='build INT8 TensorRT engine')
    parser.add_argument(
        '--dla_core', type=int, default=-1,
        help='id of DLA core for inference (0 ~ N-1)')
    args = parser.parse_args()

    engine = build_engine(
        args.model, args.category_num, args.int8, args.dla_core, args.verbose)
    if engine is None:
        raise SystemExit('ERROR: failed to build the TensorRT engine!')

    engine_path = '%s.trt' % args.model
    with open(engine_path, 'wb') as f:
        f.write(engine.serialize())
    print('Serialized the TensorRT engine to file: %s' % engine_path)


if __name__ == '__main__':
    main()
