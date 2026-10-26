import numpy as np
import scipy.interpolate as intp

def goodness_of_fit(self, obs_x=None, obs_y=None, weight=1.0, *args, **kwargs):
    """
    Calculates the Weighted Residual Factor (Rw) and returns a descriptive summary for the AI.
    Rw ranges from 0.0 (perfect match) to values > 1.0.
    """
    if obs_x is None:
        obs_x = getattr(self, "obs_x", None)
    if obs_y is None:
        obs_y = getattr(self, "obs_y", None)
    if obs_x is None or obs_y is None:
        return "### Fitting Quality\nNo observation data provided for scoring.", None

    if len(self._score) == 0 or "Rw" not in self._score:
        # 1. Interpolate simulated values onto observed grid
        sim_y = intp.interp1d(self.r, self.Gr, bounds_error=False, fill_value=0.0)(obs_x)
        
        # 2. Compute Rw metric
        diff_sq = weight * np.square(obs_y - sim_y)
        obs_sq = weight * np.square(obs_y)
        
        sum_diff = np.average(diff_sq)
        sum_obs = np.average(obs_sq)
        
        # Standard crystallographic Rw
        rw = float(np.sqrt(sum_diff / sum_obs) if sum_obs > 0 else 1.0)
    else:
        rw = self._score.get("Rw", None)

    # 3. Assign crystallographic evaluation label
    if rw < 0.10:
        label = "Excellent (Well-refined)"
    elif rw < 0.20:
        label = "Good (Reasonable fit)"
    elif rw < 0.35:
        label = "Fair (Good framework, light atom noise)"
    elif rw < 0.65:
        label = "Sublattice Matched (Heavy atom skeleton OK)"
    else:
        label = "Unacceptable (Gross structural mismatch)"

    # 4. Format as Markdown report
    report = [
        "### Overall Fitting Quality (Rw)",
        f"- **Weighted Residual Factor (Rw)**: {rw:.4f}",
        f"- **Evaluation**: {label}",
        f"- **Status**: The residual mismatch is {rw*100:.1f}% of the experimental signal magnitude."
    ]
    return "\n".join(report), rw


def goodness_of_fit_2(self, obs_x=None, obs_y=None, *args, **kwargs):
    """
    Calculates the overall fitting score and returns a descriptive summary for the AI.
    The score ranges from 0.0 to 1.0 (perfect match).
    """
    if obs_x is None:
        obs_x = getattr(self, "obs_x", None)
    if obs_y is None:
        obs_y = getattr(self, "obs_y", None)
    if obs_x is None or obs_y is None:
        return "### Fitting Score\nNo observation data provided for scoring.", None

    # Interpolate simulated values onto observed grid
    sim_y = intp.interp1d(self.r, self.Gr, bounds_error=False, fill_value=0.0)(obs_x)
    
    # Compute 1.0 - Rw score
    diff = np.sum((obs_y - sim_y)**2)
    norm = np.sum(obs_y**2)
    score = float(1.0 - np.sqrt(diff / norm) if norm > 0 else 0.0)

    # Assign evaluation label
    if score > 0.95:
        label = "Excellent (Well-refined)"
    elif score > 0.85:
        label = "Good (Reasonable fit)"
    elif score > 0.70:
        label = "Fair (Needs structural adjustment)"
    else:
        label = "Poor (Potential model mismatch)"

    # Format as Markdown report
    report = [
        "### Overall Fitting Score",
        f"- **Score**: {score:.4f} / 1.0000",
        f"- **Evaluation**: {label}",
        f"- **Status**: The model accounts for {score*100:.1f}% of the experimental PDF signal."
    ]
    return "\n".join(report), score
    

