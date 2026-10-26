
## LLM 
from .explain import params, reasoning, scoring, visualize


def explain(self, *args, **kwargs):
    """
    Make reports for LLM/VLM:
    """

    if self.obs_x is not None:
        kwargs["obs_x"] = self.obs_x
            
    if self.obs_y is not None:
        kwargs["obs_y"] = self.obs_y

    reports = {}
    reports["simulation_conditions"] = {
        "report":params.simulation(self, *args, **kwargs),
        "type":"conditions",
        "dtype": "text"
        }
    reports["peak_contributions"] = {
        "report":reasoning.peak_contributions(self, *args, **kwargs),
        "type":"results",
        "dtype": "text"
        }
    reports["simulation_image"] = {
        "report":visualize.simulation_image(self, *args, **kwargs),
        "type":"results",
        "dtype": "image"
        }

    gof_text, score = scoring.goodness_of_fit(self, *args, **kwargs)
    reports["goodness_of_fit"] = {
        "report":gof_text,
        "metadata":{"gof_score":score},            
        "type":"results",
        "dtype": "text"
        }
    return reports
