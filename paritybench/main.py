import argparse
import logging
import os
import pandas as pd
import pickle
import sys
import time
import torch
import torch._dynamo
from functools import partial

from paritybench.crawler import CrawlGitHub
from paritybench.evaluate import evaluate_all, evaluate_pyfile_subproc
from paritybench.compile import compile_all, compile_pyfile_subproc
from paritybench.generate import generate_all, generate_zipfile_subproc
from paritybench.generate import write_helpers
from paritybench.utils import subproc_wrapper, tempdir_wrapper, get_opset

log = logging.getLogger(__name__)

NATIVE_FUNCTION_YAML_PATH = f"/data/users/{os.environ['USER']}/fbsource/xplat/caffe2/aten/src/ATen/native/native_functions.yaml"
TAGS_YAML_PATH = f"/data/users/{os.environ['USER']}/fbsource/xplat/caffe2/aten/src/ATen/native/tags.yaml"
PORTABLE_YAML_PATH = f"/data/users/{os.environ['USER']}/fbsource/xplat/executorch/kernels/portable/functions.yaml"

def main_one_file(fn, path, args):
    if ':' in path and not args.filter:
        path, args.filter = path.split(':', 2)
    assert os.path.isfile(path) or os.path.isdir(path)

    fn = partial(fn, args=args)

    if not args.no_fork:
        wrapper = subproc_wrapper
    else:
        wrapper = tempdir_wrapper

    errors, stats = wrapper(path, fn=fn)

    errors.print_report()
    log.info(f"Stats: {stats}")
    return

def get_args(raw_args=None):
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--download", action="store_true", help="[SLOW:days] crawl and download top github projects")
    group.add_argument("--generate-all", action="store_true",
                       help="Turn crawled github projects into generated testcases")
    group.add_argument("--generate-one", "-g", help="Process a .zip file from a github download")
    group.add_argument("--evaluate-one", "-e", help="Check torch.jit.script on a given test_*.py file")
    group.add_argument("--compile-one", help="Check torch.jit.script on a given test_*.py file")
    group.add_argument("--compile-all", action="store_true", help="Check torch.jit.script on a given test_*.py file")
    group.add_argument("--evaluate-all", action="store_true", help="Check torch.jit.script parity")
    group.add_argument("--load-pickle", action="store_true", help="Load saved stats")

    parser.add_argument("--verbose", action="store_true", help="Print more logs")

    parser.add_argument("--jobs", "-j", type=int, default=4)
    parser.add_argument("--offset", type=int, default=0, help="Pick files starting from this offset. Together with --limit, we can run through all files in multiple separate runs")
    parser.add_argument("--limit", "-l", type=int, help="only run the first N files")
    parser.add_argument("--filter", "-f", "-k", help="only run module containing given name")
    parser.add_argument("--no-fork", action="store_true", help="don't run *-one test in a subprocess")
    parser.add_argument("--memory-limit-gb", type=int, default=10)

    parser.add_argument("--onnxdir", type=str, help="dir where to export modules to onnx during evaluate")
    parser.add_argument("--fullgraph", default=False, action="store_true", help="use fullgraph(no python fall back) when compiling with dynamo")
    parser.add_argument("--compile_mode", default="dynamo", type=str, help="choose a mode of compilation: dynamo, export or torchscript")
    parser.add_argument("--backend", default="inductor", type=str, help="dynamo backends: {}".format(torch._dynamo.list_backends()))
    parser.add_argument("--device", default="cuda", type=str, help="evaluate modules using cuda or cpu")
    parser.add_argument("--download-dir", default="./paritybench_download", help="dir where to download project default: ./paritybench_download")
    parser.add_argument("--tests-dir", default="./generated", help="dir where to generate test scripts default: ./generated")
    parser.add_argument("--metric-path", type=str, help="path of the compilation metric")

    parser.add_argument("--pickle-path", type=str, help="path of the pickle file")

    args = parser.parse_args(raw_args)
    return args

def main(raw_args=None):
    assert sys.version_info >= (3, 8), "Python 3.8+ required, got: {}".format(sys.version)
    logging.basicConfig(level=logging.INFO)
    args = get_args(raw_args)

    os.environ["RLIMIT_AS_GB"] = str(args.memory_limit_gb)

    if args.download:
        return CrawlGitHub(args.download_dir, max_count=args.limit).download()

    write_helpers()
    # generate mode doesn't work well with `spawn`
    if not args.generate_one and not args.generate_all:
        torch.multiprocessing.set_start_method('spawn')

    if args.evaluate_one:
        return main_one_file(evaluate_pyfile_subproc, args.evaluate_one, args)

    if args.compile_one:
        opset = get_opset(NATIVE_FUNCTION_YAML_PATH, TAGS_YAML_PATH, PORTABLE_YAML_PATH)
        if args.verbose:
            print(sorted(opset))
        callfn = partial(compile_pyfile_subproc, opset=opset)
        return main_one_file(callfn, args.compile_one, args)

    if args.compile_all:
        opset = get_opset(NATIVE_FUNCTION_YAML_PATH, TAGS_YAML_PATH, PORTABLE_YAML_PATH)
        if args.verbose:
            print(sorted(opset))
        return compile_all(opset, args, tests_dir=args.tests_dir, offset=args.offset, limit=args.limit, jobs=args.jobs)

    if args.load_pickle:
        with open(args.pickle_path, "rb") as f:
            stats = pickle.load(f)

            index = ("projects", "tests", "ops")
            report = pd.DataFrame(
                [[stats[f"{k}"], stats[f"{k}_passed"], "{:.1%}".format(stats[f"{k}_passed"] / (stats[f"{k}"] or 1))]
                for k in index],
                index=index,
                columns=["total", "passing", "score"],
            )

            log.info(f"TOTAL: {stats}\n\nParityBench:\n{report}\n\n")

            for key, value in sorted(stats.items(), key=lambda x: x[1], reverse=True):
                if key.startswith("aten"):
                    print(f"{key}: {value}")

            print("\n\n")

            for key, value in sorted(stats.items(), key=lambda x: x[1], reverse=True):
                if key.startswith("set("):
                    print(f"{key}: {value}")

        return

    if args.generate_one:
        return main_one_file(generate_zipfile_subproc, args.generate_one, args)

    if args.generate_all:
        return generate_all(args, download_dir=args.download_dir, limit=args.limit, jobs=args.jobs)

    # args.evaluate_all is the default:
    return evaluate_all(args, tests_dir=args.tests_dir, offset=args.offset, limit=args.limit, jobs=args.jobs)
