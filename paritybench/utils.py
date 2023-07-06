from collections import OrderedDict
import copy
import functools
import logging
import numpy as np
import os
import platform
import random
import re
import resource
import signal
import sys
import tempfile
import time
import types
import torch
import yaml

from torch import multiprocessing
from torch._dynamo.utils import clone_inputs
from torch.utils._pytree import tree_map

from torchgen.gen import parse_native_yaml

from paritybench.reporting import ErrorAggregatorDict, Stats

log = logging.getLogger(__name__)


def call_with_timeout(fn, args, kwargs=None, timeout=10):
    kwargs = kwargs or {}
    parent_conn, child_conn = multiprocessing.Pipe()
    start = time.time()
    proc = multiprocessing.Process(target=call_with_timeout_subproc, args=(fn, args, kwargs, child_conn))
    proc.start()
    while proc.is_alive():
        if parent_conn.poll(1):
            result = parent_conn.recv()
            proc.join()
            return result
        if time.time() - start > timeout:
            os.kill(proc.pid, signal.SIGINT)  # maybe generate a stack trace for debugging
            time.sleep(1)
            proc.terminate()
            proc.join(10)
            raise TimeoutError(f"took longer than {timeout} seconds")

    proc.join()
    if proc.exitcode == 0:
        return parent_conn.recv()
    else:
        raise OSError(f"exitcode should be 0, got {proc.exitcode}")


def call_with_timeout_subproc(fn, args, kwargs, return_pipe):
    use_rlimit = (
        os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") // 1024 ** 3 < 1000
        if platform.system() == "Linux"
        else True
    )
    if use_rlimit:
        _, hard = resource.getrlimit(resource.RLIMIT_AS)
        resource.setrlimit(resource.RLIMIT_AS, (int(os.environ.get("RLIMIT_AS_GB", 10)) * 1024 ** 3, hard))
    try:
        result = fn(*args, *kwargs)
        return_pipe.send(result)
    except Exception:
        log.exception("Error from subprocess")
        sys.exit(1)


def import_file(path):
    """
    :param path: to a *.py file
    :return: a python module
    """
    module = types.ModuleType(re.findall(r"test_[^.]+", path)[0])
    sys.modules[module.__name__] = module
    exec(compile(open(path).read(), filename=path, mode='exec'),
         module.__dict__, module.__dict__)
    if not hasattr(module, "TESTCASES"):
        module.TESTCASES = []
    return module


def subproc_wrapper(path: str, fn: callable, timeout: int = 900):
    """
    A wrapper around call_with_timeout() adding a temp dir and error handling.

    :param path: path to code to test
    :param fn: function to run in subprocess
    :param timeout: seconds to wait
    :return: errors, stats
    """
    log.info(f"Running {path}")
    with tempfile.TemporaryDirectory(prefix="paritybench") as tempdir:
        try:
            return call_with_timeout(fn, (tempdir, path), {}, timeout=timeout)
        except TimeoutError:
            return ErrorAggregatorDict.single(
                "meta",
                TimeoutError("Timeout testing module"),
                path
            ), Stats({"timeout": 1})
        except OSError:
            return ErrorAggregatorDict.single(
                "meta",
                OSError("Crash testing module"),
                path
            ), Stats({"crash": 1})


def tempdir_wrapper(path: str, fn: callable):
    """ Non-forking version of subproc_wrapper """
    log.info(f"Running {path}")
    with tempfile.TemporaryDirectory(prefix="paritybench") as tempdir:
        return fn(tempdir, path)


def wrap_args(args, device="cuda"):
    device = torch.device(device)
    return [x.to(device) if isinstance(x, torch.Tensor) else x for x in copy.deepcopy(args)]


def wrap_kwargs(kwargs, device="cuda"):
    device = torch.device(device)
    wrapped_kwargs = {}
    for k, v in kwargs.items():
        if isinstance(v, torch.Tensor):
            wrapped_kwargs.update({k: v.clone().to(device)})
        else:
            wrapped_kwargs.update({k: copy.deepcopy(v)})
    return wrapped_kwargs


def get_skiplist(main_args):
    if main_args.compile_mode == 'export':
        return SKIP.get("export")
    else:
        return SKIP.get(main_args.backend)


def get_tol(main_args):
    if main_args.backend == 'inductor':
        return INDUCTOR_TOL
    else:
        return DYNAMO_TOL


@functools.lru_cache(None)
def patch_torch_manual_seed():
    """Make torch manual seed deterministic. Helps with accuracy testing."""

    def deterministic_torch_manual_seed(*args, **kwargs):
        from torch._C import default_generator

        seed = 1337
        import torch.cuda

        if not torch.cuda._is_in_bad_fork():
            torch.cuda.manual_seed_all(seed)
        return default_generator.manual_seed(seed)

    torch.manual_seed = deterministic_torch_manual_seed


def reset_rng_state():
    torch.manual_seed(1337)
    random.seed(1337)
    np.random.seed(1337)


def cast_to(dtype, model, inputs):
    # cast model and inputs to fp16
    if dtype == torch.float16:
        model = model.half()
    else:
        model = model.to(dtype)

    inputs = tree_map(
        lambda x: x.to(dtype)
        if isinstance(x, torch.Tensor) and x.is_floating_point()
        else x,
        inputs,
    )
    return model, inputs


def cast_to_fp64(model, inputs):
    return cast_to(torch.float64, model, inputs)


def get_cosine_and_fp64_outputs(model, example_inputs):
    # Collect the fp64 reference outputs to be used later for accuracy checking.
    fp64_outputs = None
    cosine = False
    reset_rng_state()
    try:
        model_fp64, inputs_fp64 = cast_to_fp64(
            copy.deepcopy(model),
            clone_inputs(example_inputs),
        )
        fp64_outputs = model_fp64(inputs_fp64)
    except Exception:
        log.warning(
            "fp64 golden ref were not generated. Setting accuracy check to cosine",
        )
        cosine = True
        fp64_outputs = None
    return cosine, fp64_outputs

def get_core_aten_ops(native_function_yaml_path, tags_yaml_path):
    parsed_yaml = parse_native_yaml(native_function_yaml_path, tags_yaml_path)
    native_functions = parsed_yaml.native_functions

    core_op_names = set()
    for function in native_functions:
        if "core" in function.tags:
            core_op_names.add(str(function.func.name))
            if function.structured_delegate is not None:
                core_op_names.add(str(function.structured_delegate))
            if len(function.autogen) > 0:
                for autogen_func in function.autogen:
                    core_op_names.add(str(autogen_func))

    # Manually add some core operators that won't be captured by the above
    core_op_names.add("div.Scalar_out")
    core_op_names.add("mul.Scalar_out")
    core_op_names.add("add.Scalar_out")
    core_op_names.add("sub.Scalar_out")

    return core_op_names

def get_portable_ops(portable_yaml_path):
    parsed_yaml = ""
    with open(portable_yaml_path, "r") as stream:
        try:
            parsed_yaml = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    portable_ops = set()
    for op_entry in parsed_yaml:
        if 'op' in op_entry:
            portable_ops.add(op_entry["op"])

    return portable_ops

def add_op_to_opset(function, opset):
    op_name = str(function.func.name)
    opset.add("aten::{}".format(op_name))

    # Add structured delegates
    if function.structured_delegate is not None:
        opset.add("aten::{}".format(str(function.structured_delegate)))
    # Add autogen
    if len(function.autogen) > 0:
        for autogen_func in function.autogen:
            opset.add("aten::{}".format(str(autogen_func)))

def get_opset(native_function_yaml_path, tags_yaml_path, portable_yaml_path):
    core_op_names = get_core_aten_ops(native_function_yaml_path, tags_yaml_path)
    portable_op_names = get_portable_ops(portable_yaml_path)

    parsed_yaml = parse_native_yaml(native_function_yaml_path, tags_yaml_path)
    native_functions = parsed_yaml.native_functions

    opset = set()
    for function in native_functions:
        if "core" in function.tags:
            add_op_to_opset(function, opset)
        else:
            if function.structured_delegate is not None:
                out_var = str(function.structured_delegate)
                if out_var in portable_op_names or out_var in core_op_names:
                    add_op_to_opset(function, opset)
            if len(function.autogen) > 0:
                for autogen_func in function.autogen or autogen_func in core_op_names:
                    if str(autogen_func) in portable_op_names:
                        add_op_to_opset(function, opset)

    # Some ops need to be handled explicitly
    opset.add("aten::mean")

    return opset

DYNAMO_TOL = 1e-4
INDUCTOR_TOL = 1e-3

SKIP_DYNAMO_EAGER = [
    "./generated/test_deepinsight_insightface.py:deeplab_xception_transfer_basemodel",
    "./generated/test_SforAiDl_vformer.py:PVTClassification",
    "./generated/test_SforAiDl_vformer.py:PVTDetection",  # try ... catch ...
    "./generated/test_BlinkDL_RWKV_LM.py:RWKV_ChannelMix",  # Subclasses torch.jit.ScriptModule
    "./generated/test_facebookresearch_pytext.py:ContextualTokenEmbedding",
    "./generated/test_pytorch_translate.py:OutputProjection",  # shape_as_tensor
    "./generated/test_ludwig_ai_ludwig.py:_VectorPreprocessing",
    "./generated/test_ludwig_ai_ludwig.py:_DatePreprocessing",  # torch.jit.isinstance
    "./generated/test_adapter_hub_adapter_transformers.py:GPTNeoSelfAttention", # torch.where with dtype torch.uint8 is now deprecated.
    "./generated/test_ZhaoJ9014_face_evoLVe.py:AM_Softmax",
    "./generated/test_ZhaoJ9014_face_evoLVe.py:CircleLoss",
    "./generated/test_ZhaoJ9014_face_evoLVe.py:MagFace",
    "./generated/test_fangchangma_self_supervised_depth_completion.py:PhotometricLoss",  # torch.index with dtype torch.uint8 is now deprecated.
    "./generated/test_megvii_research_NAFNet.py:PSNRLoss",
    "./generated/test_murufeng_FUIR.py:PSNRLoss",
    "./generated/test_swz30_Restormer.py:PSNRLoss",  # Dynamo convert numpy.float64 to python float32.
]
SKIP_INDUCTOR = [
    "./generated/test_BIGBALLON_CIFAR_ZOO.py:SKConv",
    "./generated/test_CDOTAD_AlphaGAN_Matting.py:_AsppBlock",
    "./generated/test_CDOTAD_AlphaGAN_Matting.py:Encoder",
    "./generated/test_LikeLy_Journey_SegmenTron.py:_GlobalAvgPooling",
    "./generated/test_Minerva_J_Pytorch_Segmentation_multi_models.py:ChainedResidualPoolImproved",
    "./generated/test_NVIDIA_CleanUNet.py:CleanUNet",
    "./generated/test_VITA_Group_DeblurGANv2.py:ConvBlock",
    "./generated/test_Tramac_awesome_semantic_segmentation_pytorch.py:_GlobalAvgPooling",
    "./generated/test_Tramac_awesome_semantic_segmentation_pytorch.py:XceptionA",
    "./generated/test_alibaba_EasyCV.py:ClassAttentionBlock",
    "./generated/test_arnab39_cycleGAN_PyTorch.py:ResidualBlock",
    "./generated/test_chengchunhsu_EveryPixelMatters.py:Bottleneck",
    "./generated/test_deepinsight_insightface.py:DeepLabv3_plus",
    "./generated/test_deepinsight_insightface.py:Xception",
    "./generated/test_egorzakharov_PerceptualGAN.py:ResBlock",
    "./generated/test_fidler_lab_defgrid_release.py:PPM",
    "./generated/test_iscyy_yoloair.py:BottleneckCSPDNL",
    "./generated/test_iscyy_yoloair.py:DNL",
    "./generated/test_jfzhang95_pytorch_deeplab_xception.py:InvertedResidual",
    "./generated/test_lescientifik_open_brats2020.py:UBlock",
    "./generated/test_ltkong218_IFRNet.py:ResBlock",
    "./generated/test_nihalsid_ViewAL.py:InvertedResidual",
    "./generated/test_sithu31296_semantic_segmentation.py:PSAP",
    "./generated/test_voldemortX_pytorch_auto_drive.py:SpatialConv",
    "./generated/test_yassouali_pytorch_segmentation.py:_PSPModule",  # Backward fails.
]
SKIP = {
    "eager": SKIP_DYNAMO_EAGER,
    "inductor": SKIP_DYNAMO_EAGER + SKIP_INDUCTOR,
    "export": SKIP_DYNAMO_EAGER,
}
