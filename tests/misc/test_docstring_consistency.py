import os
import sys
import pkgutil
import importlib
import unittest
import numpy as np

# --- Ensure project root is on sys.path so LOCAL specsuite loads ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)
# -------------------------------------------------------------------

from get_function_info import get_module_function_data
import specsuite


# ----------------------------- Helpers -------------------------------- #

TYPE_MAP = {
    "int": int,
    "float": float,
    "str": str,
    "bool": bool,
    "list": list,
    "dict": dict,
    "tuple": tuple,
    "np.ndarray": np.ndarray,
    "numpy.ndarray": np.ndarray,
}


def iter_specsuite_modules():
    """Yield fully-qualified module names inside specsuite/."""
    for info in pkgutil.walk_packages(specsuite.__path__, prefix="specsuite."):
        if not info.name.endswith("__init__"):
            yield info.name


def strip_kwargs(names, types=None):
    """Remove 'kwargs' from argument lists while preserving alignment."""
    if "kwargs" not in names:
        return names, types

    idx = names.index("kwargs")
    names = names[:idx]
    if types is not None:
        types = types[:idx]
    return names, types


def parse_doc_type(text):
    """
    Convert docstring type notation ("int" or "int | str")
    into Python types or lists of types.
    """
    if "|" in text:
        return [TYPE_MAP[t.strip()] for t in text.split("|")]
    return TYPE_MAP[text]


def extract_module_data(module_name):
    """Import a module and return its parsed function metadata."""
    module = importlib.import_module(module_name)
    return get_module_function_data(module)


# ------------------------------ Tests --------------------------------- #

class TestDocstringConsistency(unittest.TestCase):

    def test_parameter_name_agreement(self):
        """Ensure docstring parameter NAMES match function signature."""
        for module_name in iter_specsuite_modules():
            functions = extract_module_data(module_name)

            for fn in functions:
                if not fn["docstring"]:
                    continue

                arg_names = [a["name"] for a in fn["arguments"]]
                doc_names = [p["name"] for p in fn["docstring"]["parameters"]]

                arg_names, _ = strip_kwargs(arg_names)

                self.assertEqual(
                    arg_names, doc_names,
                    msg=f"Name mismatch in {module_name}.{fn['name']}\n"
                        f"  args: {arg_names}\n"
                        f"  docs: {doc_names}"
                )

    def test_annotations_exist(self):
        """Ensure every function argument has a type annotation."""
        for module_name in iter_specsuite_modules():
            functions = extract_module_data(module_name)

            for fn in functions:
                if not fn["docstring"]:
                    continue

                arg_names = [a["name"] for a in fn["arguments"]]
                arg_types = [a["annotation"] for a in fn["arguments"]]

                arg_names, arg_types = strip_kwargs(arg_names, arg_types)

                # None means missing annotation
                self.assertNotIn(
                    None, arg_types,
                    msg=f"Missing type annotation in {module_name}.{fn['name']}"
                )

    def test_type_agreement(self):
        """Ensure docstring-declared types match python type annotations."""
        for module_name in iter_specsuite_modules():
            functions = extract_module_data(module_name)

            for fn in functions:
                if not fn["docstring"]:
                    continue

                arg_names = [a["name"] for a in fn["arguments"]]
                arg_types = [a["annotation"] for a in fn["arguments"]]
                doc_types = [p["type"] for p in fn["docstring"]["parameters"]]

                arg_names, arg_types = strip_kwargs(arg_names, arg_types)

                for doc_type, annot in zip(doc_types, arg_types):
                    expected = parse_doc_type(doc_type)

                    if isinstance(expected, list):
                        annot = list(annot.__args__)

                    self.assertEqual(
                        expected, annot,
                        msg=f"Type mismatch in {module_name}.{fn['name']}\n"
                            f"  expected: {expected}\n"
                            f"  annot:    {annot}"
                    )


if __name__ == "__main__":
    unittest.main()
