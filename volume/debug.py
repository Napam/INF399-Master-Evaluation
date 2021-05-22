from typing import Any, Callable, Union
import inspect
import re

def _tag(obj: Any, fname: str, offset: int = 0) -> str:
    """
    Obtain tag: (linenumber, function)
    fname: funcname
    offset: stack offset, stack frame index is 1 by default (prev stack)
    """
    frames = inspect.stack()

    # frames[0] is current frame,
    # frames[1] is previous
    # frames[-1] is first frame
    # remember stacks are LIFO
    # index 2 means 2+1 stack frames back

    frame = frames[1 + offset]
    funcname = frame.function
    lineno = frame.lineno
    pre_code = frame.code_context[0]
    arg = re.findall(rf"{fname}\((.*)\)", pre_code)[0]

    f_locals = frame.frame.f_locals

    tags = [lineno, funcname]
    if "self" in f_locals:  # Relies on convention
        classname = f_locals["self"].__class__.__name__
        tags[1] = classname + "." + funcname

    tags = f"\033[32m{tuple(tags)}\033[0m ".replace("'", "")

    return tags + f"{arg}"


def debug(obj: Any):
    """
    Tag and print any Python object
    """
    tag = _tag(obj, "debug", 1)
    strobj = str(obj)
    if "\n" in strobj:
        # so __str__ representation dont get wrecked if it
        # spans multiple lines
        strobj = "\n" + strobj
        strobj = strobj.replace("\n", "\n\t")
    print(f"{tag}: {strobj}")


def debugs(tensor: Any):
    """
    Tag and print shape of thing that has .shape (torch tensors, ndarrays, tensorflow tensors, ...)
    """
    tag = _tag(tensor.shape, "debugs", 1)
    print(f"{tag}: {str(tensor.shape)}")


def debugt(obj: Any):
    """
    Tag and print type, if has __len__, print that too
    """
    tag = _tag(type(obj), "debugt", 1)
    info = f"{tag}: {type(obj)}"
    try:
        info += f", len: {len(obj)}"
    except TypeError:
        pass
    print(info)