"""General use decorators for easier readability"""

import time
from typing import Callable


class Progress:
    """Decorator functions for tracking script progress"""

    @staticmethod
    def details(func: Callable):
        """Gives details on overall script progress

        Args:
            func (Callable): main function in smTIRF.py file

        Returns:
            wrapper (Callable): inner wrapper function
        """

        def wrapper(*args, **kwargs):
            """Inner wrapper function"""
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
        """Tracks only start and finish of function

        Args:
            action (str): action taking place
            ending (str): ending of descriptive string
            indent (int, optional): how far to indent print statement. Defaults to 0.
        """

        def inner(func: Callable):
            """Inner function of decorator

            Args:
                func (Callable): function acted upon
            """

            def wrapper(*args, **kwargs):
                """Inner wrapper function"""
                print(f"{'   '*indent}Starting {action} {ending}")
                func(*args, **kwargs)
                print(f"{'   '*indent}Finished {action} {ending}", end="\n" * 2)

            return wrapper

        return inner
