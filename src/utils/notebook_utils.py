def in_notebook():
    """
    Checks if code is running in IPython/Jupyter notebook versus script on command line.
    Returns:
        `True` if function runs in IPython/Juypter kernel.
        `False` if function run in python interperter on command line.
    """
    import builtins

    # The name __IPYTHON__ is defined in Jupyter or IPython but not a basic Python interpreter.
    return hasattr(builtins, "__IPYTHON__")
