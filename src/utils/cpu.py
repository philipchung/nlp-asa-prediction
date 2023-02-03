from multiprocessing import cpu_count


def num_cpu(n_cpu):
    "Return all CPUs available if `None` or `-1` specified, else passthrough"
    if n_cpu is None:
        return cpu_count()
    elif n_cpu == -1:
        return cpu_count()
    else:
        return n_cpu
