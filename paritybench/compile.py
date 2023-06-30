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
from torch._decomp import core_aten_decompositions
from torch._dynamo.testing import same
from torch._export import ExportDynamoConfig

from paritybench.reporting import ErrorAggregatorDict, Stats
from paritybench.utils import import_file, get_skiplist, get_cosine_and_fp64_outputs, get_tol, \
    patch_torch_manual_seed, reset_rng_state, subproc_wrapper, wrap_args, wrap_kwargs

from torch._ops import OpOverload

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


def compile_nn_module(opset, nn_cls, get_init_args, get_forward_args, record_error, main_args, path):
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

    try:
        DECOMP_TABLE = core_aten_decompositions()

        with torch._dynamo.config.patch(dataclasses.asdict(ExportDynamoConfig())):
            exported_model, _ = torch._dynamo.export(
                nn,
                *args,
                aten_graph=True,
                tracing_mode="symbolic",
                decomposition_table=DECOMP_TABLE,
                constraints=None,
                assume_static_by_default=True,
                **kwargs
            )

            ops = set()
            for node in exported_model.graph.nodes:
                if node.op == "call_function" and isinstance(
                    node.target, (OpOverload)
                ):
                    ops.add(node.target.name())

            not_supported = set()
            for op in ops:
                if op not in opset:
                    not_supported.add(op)

            if main_args.verbose:
                print("ops found: {}".format(ops))
                print("not supported: {}".format(not_supported))

    except Exception as e:
        record_error('run_jit {} '.format(main_args.compile_mode), e)
        raise JitFailed()

    all_supported = len(not_supported) == 0
    return True, all_supported, not_supported


def compile_pyfile_subproc(tempdir: str, path: str, args, opset):
    """
    compile/test all the TESTCASES in path.

    :param path: *.py file to test
    :return: errors, stats
    """
    errors = ErrorAggregatorDict(path)
    stats = Stats()
    module = import_file(path)

    if not module.TESTCASES:
        return errors, stats

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

        stats["tests"] += 1
        repro = f"{nn_cls.__name__} # pytest {path} -k test_{index:03d}"
        try:
            success, all_supported, not_supported = compile_nn_module(
                opset,
                nn_cls,
                get_init_args,
                get_forward_args,
                partial(errors.record, module=repro),
                main_args=args,
                path=path)
            stats["tests_passed"] += int(success)
            if success:
                stats["ops"] += 1
                if all_supported:
                    stats["ops_passed"] += 1
                else:
                    stats["ops_failed"] += 1

            for op in not_supported:
                stats[op] += 1

        except JitFailed:
            pass
        except EagerFailed:
            pass
        except OnnxFailed:
            pass

    stats["tests_failed"] = stats["tests"] - stats["tests_passed"]

    if stats["compile_failed"]:
        stats["projects_failed"] += 1
    else:
        stats["projects_passed"] += 1

    return errors, stats


def compile_all(opset, args, tests_dir: str = './generated', offset: int = 0, limit: int = None,
                 jobs=4):
    """
    Generate a paritybench score, main entrypoint for this module.

    :param tests_dir: directory containing paritybench testcases
    :param limit: optional maximum number of files to process
    :param fn: inner function to run the tests
    :param jobs: how many processes to run at once
    """
    feval = partial(compile_pyfile_subproc, args=args, opset=opset)
    fn = partial(subproc_wrapper, fn=feval)
    start = time.time()
    stats = Stats()
    errors = ErrorAggregatorDict()
    testfiles = [os.path.join(tests_dir, f)
                 for f in os.listdir(tests_dir)
                 if re.search(r"test_.*[.]py$", f)]
    testfiles.sort()

    if limit:
        testfiles = testfiles[offset: offset+limit]

    pool = ThreadPool(jobs)
    for errors_part, stats_part in pool.imap_unordered(fn, testfiles):
        errors.update(errors_part)
        stats.update(stats_part)
    pool.close()
    errors.print_report()
    index = ("projects", "tests", "ops")
    report = pd.DataFrame(
        [[stats[f"{k}"], stats[f"{k}_passed"], "{:.1%}".format(stats[f"{k}_passed"] / (stats[f"{k}"] or 1))]
         for k in index],
        index=index,
        columns=["total", "passing", "score"],
    )

    log.info(f"TOTAL: {stats}, took {time.time() - start:.1f} seconds\n\n{args.compile_mode} {args.backend} ParityBench:\n{report}")

    if args.pickle_path is not None:
        with open(args.pickle_path, "wb") as f:
            f.write(pickle.dumps(stats))
