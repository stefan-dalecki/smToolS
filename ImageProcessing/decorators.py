"""General use decorators for easier readability"""
import time
from typing import Callable


class Progress:
    """Decorator function(s) for tracking script progress"""

    @staticmethod
    def movie_timer(func: Callable):
        """Measures the time it takes to process a movie
        Includes time spent looking at figures
        Best used without displaying figures

        Args:
            func (Callable): Main function from script.py
        """

        def wrapper(*args, **kwargs):
            """Inner wrapper function

            Returns:
                any type of function result"""
            print("\nBeginning Movie Analysis...")
            begin = time.time()
            function_result = func(*args, **kwargs)
            end = time.time()
            print(
                "Movie Analysis Completed",
                f"Time elapsed : {(end - begin):.2f} sec",
                sep="\n",
                end="\n" * 3,
            )
            return function_result

        return wrapper

    @staticmethod
    def time_me(func: Callable):
        """Calculates the time taken to complete function call

        Args:
            func (Callable): function to measure
        """

        def wrapper(*args, **kwargs):
            """Returns result of function

            Returns:
                any type of function return
            """
            begin = time.time()
            function_result = func(*args, **kwargs)
            end = time.time()
            print(" " * 9, f"{func.__name__} Elapsed time : {(end-begin):.2f}")
            return function_result

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
                """Inner wrapper function

                Returns:
                    any type of function result"""
                print(f"{'   '*indent}Starting {action} {ending}")
                function_result = func(*args, **kwargs)
                print(f"{'   '*indent}Finished {action} {ending}", end="\n" * 2)
                return function_result

            return wrapper

        return inner
