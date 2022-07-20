"""General use decorators for easier readability"""

import time
from typing import Callable


class Progress:
    @staticmethod
    def details(func: Callable):
        def wrapper(*args, **kwargs):
            begin = time.time()
            print("Beginning Movie Analysis...", end="\n" * 2)
            func(*args, **kwargs)
            end = time.time()
            print(
                "Movie Analysis Completed",
                f"Time elapsed : {(end - begin):.2f} sec",
                sep="\n",
                end="\n" * 3,
            )

        return wrapper

    @staticmethod
    def start_finish(*, indent: int = 0, action: str, ending: str):
        def inner(func: Callable):
            def wrapper(*args, **kwargs):
                print(f"{'   '*indent}Starting {action} {ending}")
                func(*args, **kwargs)
                print(f"{'   '*indent}Finished {action} {ending}", end="\n" * 2)

            return wrapper

        return inner
