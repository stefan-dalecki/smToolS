
from simo_tools import constants as cons
from simo_tools import trajectory as traj
from simo_tools.analysis import cutoffs as cuts


class TestBrightness:
    """
    cutoffs.Brightness.
    """

    test_class = cuts.Brightness

    def test_manual(self, trajectories_obj: traj.Trajectories):
        """
        cutoffs.Brightness.manual.
        """
        cutoff = self.test_class(method=cons.CutoffMethods.MANUAL, min=2.5, max=10)
        actual_result = cutoff.threshold(trajectories_obj)
        cutoff.display()
        assert len(actual_result) == 3
