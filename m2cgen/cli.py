"""CLI for m2cgen.

Example usage:
    $ m2cgen <path_to_file> --language java --class_name MyModel --package_name foo.bar.baz
    $ m2cgen --language java < <path_to_file>

Model can also be piped:
    # cat <path_to_file> | m2cgen --language java
"""
import sys
from argparse import ArgumentParser, FileType
from inspect import signature

import numpy as np

import m2cgen

LANGUAGE_TO_EXPORTER = {
    "python": (m2cgen.export_to_python, ["indent", "function_name"]),
    "java": (m2cgen.export_to_java, ["indent", "class_name", "package_name", "function_name"]),
    "javascript": (m2cgen.export_to_javascript, ["indent", "function_name"]),
    "visual_basic": (m2cgen.export_to_visual_basic, ["module_name", "indent", "function_name"]),
    "c_sharp": (m2cgen.export_to_c_sharp, ["indent", "class_name", "namespace", "function_name"]),
    "powershell": (m2cgen.export_to_powershell, ["indent", "function_name"]),
    "php": (m2cgen.export_to_php, ["indent", "function_name"]),
    "rust": (m2cgen.export_to_rust, ["indent", "function_name"]),
    "sql": (m2cgen.export_to_sql,
            ["indent", "function_name", "dialect", "feature_names"]),
}


# The maximum recursion depth is represented by the maximum int32 value.
MAX_RECURSION_DEPTH = np.iinfo(np.intc).max


parser = ArgumentParser(
    prog="m2cgen",
    description="Generate code in native language for provided model.")
parser.add_argument(
    "infile",
    type=FileType("rb"),
    nargs="?",
    default=sys.stdin.buffer,
    help="File with pickle representation of the model.")
parser.add_argument(
    "--language", "-l",
    type=str,
    choices=LANGUAGE_TO_EXPORTER.keys(),
    help="Target language.",
    required=True)
parser.add_argument(
    "--function_name", "-fn",
    dest="function_name",
    type=str,
    # The default value is conditional and will be set in the argument's
    # post-processing, based on the signature of the `export` function
    # that belongs to the specified target language.
    default=None,
    help="Name of the function in the generated code.")
parser.add_argument(
    "--class_name", "-cn",
    dest="class_name",
    type=str,
    help="Name of the generated class (if supported by target language).")
parser.add_argument(
    "--package_name", "-pn",
    dest="package_name",
    type=str,
    help="Package name for the generated code (if supported by target language).")
parser.add_argument(
    "--module_name", "-mn",
    dest="module_name",
    type=str,
    help="Module name for the generated code (if supported by target language).")
parser.add_argument(
    "--namespace", "-ns",
    dest="namespace",
    type=str,
    help="Namespace for the generated code (if supported by target language).")
parser.add_argument(
    "--indent", "-i",
    dest="indent",
    type=int,
    default=4,
    help="Indentation for the generated code.")
parser.add_argument(
    "--recursion-limit", "-rl",
    type=int,
    help="Sets the maximum depth of the Python interpreter stack. No limit by default",
    default=MAX_RECURSION_DEPTH)
parser.add_argument(
    "--dialect", "-d",
    type=str,
    choices=["duckdb", "mysql", "postgres"],
    default="duckdb",
    help="SQL dialect (only used with --language sql).")
parser.add_argument(
    "--feature_names", "-fnames",
    type=str,
    dest="feature_names",
    help="Comma-separated feature names used as parameter names "
         "(only used with --language sql).")
parser.add_argument(
    "--version", "-v",
    action="version",
    version=f"%(prog)s {m2cgen.__version__}")
parser.add_argument(
    "--pickle-lib", "-pl",
    type=str,
    dest="lib",
    help="Sets the lib used to save the model",
    choices=["pickle", "joblib"],
    default="pickle")


def parse_args(args):
    parsed = parser.parse_args(args)
    if parsed.feature_names is not None:
        parsed.feature_names = [
            name.strip() for name in parsed.feature_names.split(",")
            if name.strip()]
    return parsed


def generate_code(args):
    sys.setrecursionlimit(args.recursion_limit)

    with args.infile as f:
        pickle_lib = __import__(args.lib)
        model = pickle_lib.load(f)

    exporter, supported_args = LANGUAGE_TO_EXPORTER[args.language]

    kwargs = {}
    for arg_name in supported_args:
        arg_value = getattr(args, arg_name)
        if arg_value is not None:
            kwargs[arg_name] = arg_value

        # Special handling for the function_name parameter, which needs to be
        # the same as the default value of the keyword argument of the exporter
        # (this is due to languages like C# which prefer their method names to
        # follow PascalCase unlike all the other supported languages -- see
        # https://github.com/BayesWitnesses/m2cgen/pull/166#discussion_r379867601
        # for more).
        if arg_name == 'function_name' and arg_value is None:
            param = signature(exporter).parameters['function_name']
            kwargs[arg_name] = param.default

    return exporter(model, **kwargs)


def main():
    args = parse_args(sys.argv[1:])
    print(generate_code(args))
