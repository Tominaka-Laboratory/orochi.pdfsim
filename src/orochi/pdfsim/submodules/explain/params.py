import numpy as np

def simulation(self, precision=3, *args, **kwargs):
    """
    Convert Simulator settings (instrumental resolution, scattering factors, and peak shape) into a Markdown report.
    """
    phase = self.phase
    # Retrieve atomic numbers (Z) in the structure to check elemental scattering contrast
    z_values = phase.get_atomic_numbers(phase["symbol"])
    max_z_diff = max(z_values) - min(z_values) if z_values else 0

    lines = [
        "### PDF Simulation & Instrumental Parameters",
        ""
    ]

    # 1. Physical models (density and scattering factors)
    lines.append("- **Physical Models**:")

    # Density calculation model description
    density_mode = "Analytical (Assumes uniform bulk density)" if (self.phase["shape_type"] == "analytical") and (np.sum(self.phase["pbc"]) == 3) else "Detailed (Accounts for non-uniform density and Qmin; essential for complexes)"
    lines.append(f"  - [key: density] Density Calculation: {density_mode}")

    qdep = "Included (High accuracy for large Z-diff)" if self["qdepend"] else "Excluded"
    lines.append(f"  - [key: qdepend] Q-dependent scattering factors: {qdep}")

    if not self["qdepend"]:
        if max_z_diff > 30:
            lines.append(f"  - *Warning*: Large Z-difference found (ΔZ={max_z_diff}). "
                         "Ignoring Q-dependence may cause peak intensity mismatches (e.g., Ce-O or Ru-O cases).")
        elif max_z_diff > 10:
            lines.append(f"  - *Note*: Moderate Z-difference (ΔZ={max_z_diff}). Standard approximation is usually sufficient.")

    # 2. Instrumental parameters
    lines.append("- **Instrumental Resolution & Noise**:")
    lines.append(f"  - [key: qmax] Qmax: {self['qmax']:.{precision}f} Å⁻¹")
    # qdamp: Gaussian damping of PDF due to Q-resolution

    if self["qdamp"]:
        lines.append(f"  - [key: qdamp] qdamp: {self['qdamp']:.{precision}f} Å⁻¹ (Gaussian dampening from Q-resolution)")

    if self["qbroad"]:
        # qbroad: Peak broadening due to intensity noise at high Q
        lines.append(f"  - [key: qbroad] qbroad: {self['qbroad']:.{precision}f} Å⁻¹ (Broadening from intensity noise at high Q)")

    # 3. Peak shape and damping models (structural/refinement parameters)
    lines.append("- **Peak Shape & Damping Models**:")
    if self["delta1"]:
        lines.append(f"  - [key: delta1] delta1: {self['delta1']:.{precision}f} (Low-r broadening)")
    if self["delta2"]:
        lines.append(f"  - [key: delta2] delta2: {self['delta2']:.{precision}f} (r-dependent broadening)")
    if self["sratio"] != 1.0:
        lines.append(f"  - [key: sratio] sratio: {self['sratio']:.{precision}f} (Peak intensity scaling)")

    lines.append(f"  - [key: rambda] rambda: {self['rambda']:.{precision}f} (Experimental damping)")

    if self.phase["shape_type"] != "analytical":
        lines.append(f"  - [key: shape_type] Shape type: {self.phase['shape_type']}")

    # 4. Calculation range (r-range)
    lines.append(f"- **Calculation Range**: [key: rmin] {self['rmin']:.{precision}f} to [key: rmax] {self['rmax']:.{precision}f} Å")
    lines.append(f"- **Scaling Factor**: [key: scale] {self['scale']:.{precision}f}")

    return "\n".join(lines)