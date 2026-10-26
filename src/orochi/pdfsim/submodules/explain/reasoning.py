import numpy as np

def peak_contributions(self, 
                       obs_x=None, 
                       obs_y=None, 
                       top_n=12, 
                       bin_width=0.1, 
                       window=0.2, *args, **kwargs):
    """
    Extract dominant peaks and the shortest observable peak per atomic pair
    from a physics-weighted pair-distance histogram.
    """
    phase = self.phase
    all_peaks = []
    
    # Retrieve atomic numbers (Z)
    z_map = dict(zip(phase["symbol"], phase.get_atomic_numbers(phase["symbol"])))

    if obs_x is None:
        obs_x = getattr(self, "obs_x", None)
    if obs_y is None:
        obs_y = getattr(self, "obs_y", None)

    if obs_x is not None and obs_y is not None:
        sim_y = np.interp(obs_x, self.r, self.Gr)
    else:
        sim_y = None

    for n, pair_name in enumerate(self.pair_name):
        s1, s2 = pair_name.split("-")
        z_weight = z_map[s1] * z_map[s2] # Approximate weight by product of atomic numbers
        
        rij = self.split_pairs["r_ij"][n]
        cij = self.split_pairs["c_ij"][n]
        
        if len(rij) == 0:
            continue

        # 1. Aggregate by distance binning (histogram)
        bins = np.arange(self['rmin'], self['rmax'], bin_width)
        indices = np.digitize(rij, bins)
        
        for b in range(1, len(bins)):
            mask = (indices == b)
            if np.any(mask):
                r_bin = bins[b] - (bin_width / 2)
                # Physics scaling: multiply by Zi*Zj and divide by r to match G(r) weighting
                weight_sum = (np.sum(cij[mask]) * z_weight) / r_bin
                r_avg = np.average(rij[mask], weights=cij[mask])
                this_peak = {
                    "pair": pair_name,
                    "r": r_avg,
                    "intensity": weight_sum
                }

                if obs_x is not None and obs_y is not None:
                    mask = (np.array(obs_x) >= r_avg - window) & (np.array(obs_x) <= r_avg + window)
                    
                    if np.any(mask):
                        local_sim = sim_y[mask]
                        local_obs = obs_y[mask]
                        diff = np.sum((local_obs - local_sim)**2)
                        norm = np.sum(local_obs**2)
                        local_fitness = 1.0 - np.sqrt(diff / norm) if norm > 0 else 0
                    else:
                        local_fitness = 0.0

                    if local_fitness > 0.92:
                        local_match = f"{local_fitness:.3f} (Excellent)"
                    elif local_fitness > 0.85:
                        local_match = f"{local_fitness:.3f} (Good)"
                    elif local_fitness > 0.65:
                        local_match = f"{local_fitness:.3f} (Fair)"
                    else:
                        local_match = f"{local_fitness:.3f} (Poor)"

                    this_peak["local_match"] = local_match
                
                all_peaks.append(this_peak)

    if not all_peaks:
        return "No significant atomic correlations found in the current range."

    # 2. Extract shortest peak per atomic pair
    shortest_peaks = {}
    for p in sorted(all_peaks, key=lambda x: x['r']):
        if p['pair'] not in shortest_peaks:
            shortest_peaks[p['pair']] = p

    # 3. Select top N peaks by intensity and sort by r for readability
    top_peaks = sorted(all_peaks, key=lambda x: x['intensity'], reverse=True)[:top_n]
    top_peaks = sorted(top_peaks, key=lambda x: x['r'])


    # --- Build report ---
    lines = ["### Key Peak Attribution (Physics-weighted)"]
    lines.append("| Rank | Atomic Pair | r-position (Å) | Relative Intensity | Local Match |")
    lines.append("|------|-------------|----------------|--------------------|-------------|")
    
    max_int = max([p['intensity'] for p in top_peaks])
    for i, p in enumerate(top_peaks):
        if 'local_match' in p:
            lines.append(f"| {i+1} | {p['pair']} | {p['r']:.3f} | {p['intensity']/max_int:.2f} | {p['local_match']} |")
        else:   
            lines.append(f"| {i+1} | {p['pair']} | {p['r']:.3f} | {p['intensity']/max_int:.2f} |")

    lines.append("\n### First Observed Peaks (Per Pair Type)")
    lines.append("The shortest meaningful distances represented in the PDF peaks.")
    for pair_name in sorted(shortest_peaks.keys()):
        p = shortest_peaks[pair_name]
        lines.append(f"- **{pair_name}**: {p['r']:.3f} Å (Intensity: {p['intensity']/max_int:.2f})")

    return "\n".join(lines)