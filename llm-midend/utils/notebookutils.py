import IPython.display


def in_notebook() -> bool:
    try:
        shell = get_ipython().__class__.__name__
        return shell == "ZMQInteractiveShell"
    except NameError:
        return False


IN_NOTEBOOK = in_notebook()


def display(*args, **kwargs):
    if IN_NOTEBOOK:
        IPython.display.display(*args, **kwargs)
