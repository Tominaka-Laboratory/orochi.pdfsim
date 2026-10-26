import matplotlib.pyplot as plt
import numpy as np
import scipy.interpolate as intp
import io

def simulation_image(self, obs_x=None, obs_y=None, dpi=150, *args, **kwargs):
    """
    Visualize PDF simulation results (total PDF, difference curve, and partial PDFs).
    """
    # 1. Prepare common damping/correction factors
    resolution = np.exp(-1.0 * np.square(self.r * self["qdamp"]) / 2.0)
    shape_func = self.shape_function
    scale = self["scale"]
    
    # 2. Create two-panel plot (Total + Difference / Partials)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True, 
                                   gridspec_kw={'height_ratios': [1, 1]})
    
    # --- Top panel: Total G(r) ---
    if obs_x is None:
        obs_x = getattr(self, "obs_x", None)
    if obs_y is None:
        obs_y = getattr(self, "obs_y", None)

    if obs_x is not None and obs_y is not None:
        ax1.plot(obs_x, obs_y, color="blue", marker="o",
                 markersize=1.5, label="Experiment (obs)", alpha=0.4)
        ax1.plot(self.r, self.Gr, 'r-', linewidth=1.5, label="Calculation (calc)")
        
        # Compute difference curve interpolated onto observed grid
        calc_on_obs = intp.interp1d(self.r, self.Gr, bounds_error=False, fill_value=0.0)(obs_x)
        diff = obs_y - calc_on_obs
        # Offset difference curve for visual clarity
        offset = np.min(obs_y) * 1.5
        ax1.plot(obs_x, diff + offset, 'g-', linewidth=1, label="Difference")
        ax1.axhline(y=offset, color='gray', linestyle='--', linewidth=0.5)

        if len(self._score) == 0:
            rmse = np.sqrt(np.mean(np.square(diff)))
            score_label = f"RMSE: {rmse:.4f}"
        else:
            score_label_list = [
                                f"{key}:{val:.3f}" if isinstance(val, float) else f"{key}:{val}"
                                for key, val in self._score.items()
                                ]
            score_label = ", ".join(score_label_list)

    else:
        ax1.plot(self.r, self.Gr, 'r-', label="Calculation")
        score_label = "Simulation only"

    ax1.set_ylabel("G (Å⁻²)", fontsize=12)
    ax1.legend(loc='upper right', fontsize=10)
    ax1.set_title(f"Total PDF Comparison ({score_label})", fontsize=14)

    for pair, p_gr_raw in self.partial_gr.items():
        # Interpolate from internal _r grid onto output r grid
        p_gr_interp = intp.interp1d(self._r, p_gr_raw, 
                                    fill_value="extrapolate", 
                                    bounds_error=False)(self.r)
        
        # Scale partial pair distributions (without baseline density subtraction)
        p_Gr = p_gr_interp * shape_func * resolution * scale
        ax2.plot(self.r, p_Gr, label=pair, linewidth=1.2)

    ax2.set_xlabel("r (Å)", fontsize=12)
    ax2.set_ylabel("Partial G (Å⁻²)", fontsize=12)
    ax2.legend(ncol=3, loc='upper right', fontsize=9)
    ax2.set_title("Partial Pair Contributions (Zero-baseline)", fontsize=12)

    plt.tight_layout()

    # Export PNG bytes
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=dpi)
    plt.close(fig)
    return buf.getvalue()