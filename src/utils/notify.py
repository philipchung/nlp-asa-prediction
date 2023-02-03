import functools
from knockknock import teams_sender


def notify(webhook_url: str = None):
    """
    Decorator to wrap function with teams_sender if webhook_url is provided.
    If `webhook_url` argument is None, this decorator is a noop and
    will return original unwrapped function.
    """

    def decorator(f):
        @functools.wraps(f)
        def maybe_wrap_with_sender():
            if webhook_url:
                decorator_sender = teams_sender(webhook_url=webhook_url)
                wrapped_fn = decorator_sender(f)
                return wrapped_fn()
            else:
                return f()

        return maybe_wrap_with_sender

    return decorator
