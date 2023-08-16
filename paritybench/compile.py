import dataclasses
import logging
import os
import pickle
import re
import time
from functools import partial
from multiprocessing.pool import ThreadPool
import threading

import pandas as pd
import torch
import torch._dynamo
import torch._inductor
from torch.testing._internal.jit_utils import JitTestCase
from torch._decomp import core_aten_decompositions, get_decompositions
from torch._dynamo.testing import same
from torch._export import aot_export_module, ExportDynamoConfig
from torch.fx.experimental.proxy_tensor import make_fx

from paritybench.reporting import ErrorAggregatorDict, Stats
from paritybench.utils import import_file, get_skiplist, get_cosine_and_fp64_outputs, get_tol, \
    patch_torch_manual_seed, reset_rng_state, subproc_wrapper_mod, wrap_args, wrap_kwargs

from torch._ops import OpOverload
import torch.utils._pytree as pytree

log = logging.getLogger(__name__)

# Remove inductor randomness
torch._inductor.config.fallback_random = True
# Remove randomeness when torch manual seed is called
patch_torch_manual_seed()

lock = threading.Lock()


class EagerFailed(RuntimeError):
    pass

class OnnxFailed(RuntimeError):
    pass

class JitFailed(RuntimeError):
    pass

def get_module_ops(module, args, kwargs, decomp_table):
    #flat_args, _ = pytree.tree_flatten((args, kwargs))
    exported_model = make_fx(
        module,
        decomposition_table=decomp_table,
        tracing_mode="symbolic",
        _allow_non_fake_inputs=True,
    )(*args, **kwargs)

    ops = set()
    for node in exported_model.graph.nodes:
        if node.op == "call_function" and isinstance(
            node.target, (OpOverload)
        ):
            ops.add(node.target.name())

    return ops


def get_nn_module_ops(nn_cls, get_init_args, get_forward_args, record_error, main_args, path):
    """
    Run an nn.Module with torch.jit.script and see if it works the same
    as eager.

    :param nn_cls: a subclass of nn.Module to be tested
    :param get_init_args: function that returns (args, kwargs)
    :param get_forward_args: function that returns (args, kwargs)
    :param record_error: function to record an exception for debugging/reporting
    :return: True if the test passes
    """

    try:
        args, kwargs = get_init_args()
        nn = nn_cls(*args, **kwargs)
    except Exception as e:
        record_error('init', e)
        raise EagerFailed()

    device = torch.device(main_args.device)

    try:
        nn.eval()
        nn.to(device)
    except Exception:
        pass

    nn_script = None

    args, kwargs = get_forward_args()
    args = wrap_args(args, device)
    kwargs = wrap_kwargs(kwargs, device)

    scripted_ops = set()
    try:
        nn_script = torch.jit.script(nn)
        scripted_ops = set(torch.jit.export_opnames(nn_script))
        print(f"{nn_cls.__name__} scripted ops: {scripted_ops}")
    except Exception as e:
        pass

    core_decomp_ops = set()
    edge_decomp_ops = set()
    no_decomp_ops = set()

    success = True

    try:
        edge_decomp_opset = [
            torch.ops.aten.log_sigmoid_forward,
            torch.ops.aten.ones,
            torch.ops.aten.arange.default,
            torch.ops.aten.arange.start,
            torch.ops.aten.transpose,
        ]
        edge_decompositions = get_decompositions(edge_decomp_opset)

        core_decomp_ops = get_module_ops(nn, args, kwargs, core_aten_decompositions())
        edge_decomp_ops = get_module_ops(nn, args, kwargs, edge_decompositions)
        no_decomp_ops = get_module_ops(nn, args, kwargs, {})

        print(f"{nn_cls.__name__} core decomp ops: {core_decomp_ops}")
        print(f"{nn_cls.__name__} edge decomp ops: {edge_decomp_ops}")
        print(f"{nn_cls.__name__} no decomp ops: {no_decomp_ops}")

    except Exception as e:
        record_error('make_fx error from {} '.format(main_args.compile_mode), e)
        success = False

    return success, core_decomp_ops, edge_decomp_ops, no_decomp_ops, scripted_ops


def compile_pyfile_subproc(tempdir: str, path: str, args):
    """
    compile/test all the TESTCASES in path.

    :param path: *.py file to test
    :return: errors, stats
    """
    errors = ErrorAggregatorDict(path)
    stats = Stats()
    module = import_file(path)
    nn_ops_dict = {}

    if not module.TESTCASES:
        return errors, stats, nn_ops_dict

    stats["projects"] += 1

    index = -1
    for nn_cls, get_init_args, get_forward_args, compiles in module.TESTCASES:
        index += 1

        if args.filter and args.filter not in nn_cls.__name__:
            continue

        if f"{path}:{nn_cls.__name__}" in get_skiplist(args):
            continue

        # nn.module doesn't have `forward` function(e.g, has __call__ instead).
        # dynamo doesn't plan to support it yet.
        if nn_cls.forward.__name__ == "_forward_unimplemented":
            continue

        stats["export_attempts"] += 1
        repro = f"{nn_cls.__name__} # pytest {path} -k test_{index:03d}"
        try:
            success, core_decomp_ops, edge_decomp_ops, no_decomp_ops, scripted_ops = get_nn_module_ops(
                nn_cls,
                get_init_args,
                get_forward_args,
                partial(errors.record, module=repro),
                main_args=args,
                path=path)
            stats["export_attempts_passed"] += int(success)

            ops_dict = {}
            if len(core_decomp_ops) > 0:
                ops_dict["core_decomp_ops"] = core_decomp_ops
            if len(edge_decomp_ops) > 0:
                ops_dict["edge_decomp_ops"] = edge_decomp_ops
            if len(no_decomp_ops) > 0:
                ops_dict["no_decomp_ops"] = no_decomp_ops
            if len(scripted_ops) > 0:
                ops_dict["scripted_ops"] = scripted_ops

            if len(core_decomp_ops) + len(edge_decomp_ops) + len(no_decomp_ops) + len(scripted_ops) > 0:
                nn_ops_dict[f"{path}.{nn_cls.__name__}"] = ops_dict


        except JitFailed:
            pass
        except EagerFailed:
            pass
        except OnnxFailed:
            pass

    stats["export_attempts_failed"] = stats["export_attempts"] - stats["export_attempts_passed"]

    if stats["export_attempts_failed"]:
        stats["projects_failed"] += 1
    else:
        stats["projects_passed"] += 1

    return errors, stats, nn_ops_dict


def compile_all(opset, args, tests_dir: str = './generated', offset: int = 0, limit: int = None,
                 jobs=4):
    """
    Generate a paritybench score, main entrypoint for this module.

    :param tests_dir: directory containing paritybench testcases
    :param limit: optional maximum number of files to process
    :param fn: inner function to run the tests
    :param jobs: how many processes to run at once
    """
    feval = partial(compile_pyfile_subproc, args=args)
    fn = partial(subproc_wrapper_mod, fn=feval)
    start = time.time()
    stats = Stats()
    module_ops = {}
    errors = ErrorAggregatorDict()
    testfiles = [os.path.join(tests_dir, f)
                 for f in os.listdir(tests_dir)
                 if re.search(r"test_.*[.]py$", f)]
    testfiles.sort()

    if limit:
        testfiles = testfiles[offset: offset+limit]

    pool = ThreadPool(jobs)
    for errors_part, stats_part, nn_ops in pool.imap_unordered(fn, testfiles):
        errors.update(errors_part)
        stats.update(stats_part)
        module_ops.update(nn_ops)
    pool.close()
    errors.print_report()
    index = ("projects", "export_attempts")
    report = pd.DataFrame(
        [[stats[f"{k}"], stats[f"{k}_passed"], "{:.1%}".format(stats[f"{k}_passed"] / (stats[f"{k}"] or 1))]
         for k in index],
        index=index,
        columns=["total", "passing", "score"],
    )

    log.info(f"TOTAL: {stats}, took {time.time() - start:.1f} seconds\n\n{args.compile_mode} {args.backend} ParityBench:\n{report}\n\n")

    if args.pickle_path is not None:
        with open(args.pickle_path, "wb") as f:
            f.write(pickle.dumps(module_ops))
