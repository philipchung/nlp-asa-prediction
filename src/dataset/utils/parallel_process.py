from __future__ import annotations
from typing import Union, Iterable
from tqdm.auto import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count


def parallel_process(
    iterable: Union[list, Iterable],
    function: callable,
    n_jobs: int = cpu_count(),
    use_args: bool = False,
    use_kwargs: bool = False,
    desc: str = "",
):
    """
    A parallel version of the map function with a progress bar.

    Args:
        iterable (collection): An array-like or dict-like to iterate over
        function (function): A python function to apply to the elements of array.  A special requirement
            for this function is that it needs to be at top-level scope so python can pickle the function.
            Also all variables within the function must be pickle-able.  Note that only this function has
            this requirement and the calling scope for `parallel_process` can be embedded within another
            function or class.
        n_jobs (int, default=16): The number of cores to use
        use_args (boolean, default=False): Whether to consider the elements of array as tuples of arguments to function.
            Tuple elements are passed to function arguments by position.  Set this to True if function has multiple
            arguments and your tuple provides the arguments in-order.
        use_kwargs (boolean, default=False): Whether to consider the elements of array as dictionaries of
            keyword arguments to pass into function.  Set this to True if function has multiple arguments
            and you want to pass arguments to function by keyword (does not need to be in-order).
        desc (string, default=""): Description on progress bar
    Returns:
        [function(iterable[0]), function(iterable[1]), ...]
    """
    # If we set n_jobs to 1, just run a list comprehension. This is useful for benchmarking and debugging.
    if n_jobs == 1:
        if use_kwargs:
            return [function(**a) for a in tqdm(iterable)]
        elif use_args:
            return [function(*a) for a in tqdm(iterable)]
        else:
            return [function(a) for a in tqdm(iterable)]
    # Assemble the workers
    with ProcessPoolExecutor(max_workers=n_jobs) as pool:
        # Pass the elements of array into function
        if use_kwargs:
            futures = [pool.submit(function, **a) for a in iterable]
        elif use_args:
            futures = [pool.submit(function, *a) for a in iterable]
        else:
            futures = [pool.submit(function, a) for a in iterable]
        # Print out the progress as tasks complete
        kwargs = {
            "total": len(futures),
            "unit": "it",
            "unit_scale": True,
            "leave": True,
            "desc": f"{desc} (Dispatch)",
            "dynamic_ncols": True,
        }
        for f in tqdm(as_completed(futures), **kwargs):
            pass
    out = []
    # Get the results from the futures.
    for i, future in tqdm(
        enumerate(futures), desc=f"{desc} (Completed)", dynamic_ncols=True
    ):
        try:
            out += [future.result()]
        except Exception as e:
            out += [
                (
                    e,
                    f"Occurred with input element at index: {i}.",
                )
            ]
    return out
