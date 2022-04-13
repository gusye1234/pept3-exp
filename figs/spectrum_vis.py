import itertools
import math
from typing import Dict, Optional, Tuple, Union
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


COLORS = {
    "a": "#388E3C",
    "b": "orange",
    "c": "#00796B",
    "x": "orange",
    "y": "rosybrown",
    "z": "#F57C00",
    "?": "#212121",
    "f": "#212121",
    None: "#212121",
}
zorders = {
    "a": 3,
    "b": 4,
    "c": 3,
    "x": 3,
    "y": 4,
    "z": 3,
    "?": 2,
    "f": 5,
    None: 1,
}


def _annotate_ion(
    mz: float,
    intensity: float,
    annotation,
    color_ions: bool,
    annotate_ions: bool,
    annotation_kws: Dict[str, object],
    ax: plt.Axes,
    complex_anno = True,
    overlap=False
) -> Tuple[str, int]:
    """
    Annotate a specific fragment peak.
    Parameters
    ----------
    mz : float
        The peak's m/z value (position of the annotation on the x axis).
    intensity : float
        The peak's intensity (position of the annotation on the y axis).
    annotation : Optional[MoleculeFragmentAnnotation,
                          PeptideFragmentAnnotation]
        The annotation that will be plotted.
    color_ions : bool
        Flag whether to color the peak annotation or not.
    annotate_ions : bool
        Flag whether to annotation the peak or not.
    annotation_kws : Dict
        Keyword arguments for `ax.text` to customize peak annotations.
    ax : plt.Axes
        Axes instance on which to plot the annotation.
    Returns
    -------
    Tuple[str, int]
        A tuple of the annotation's color as a hex string and the annotation's
        zorder.
    """
    # No annotation -> Just return peak styling information.
    if annotation is None:
        return COLORS.get(None), zorders.get(None)
    # Else: Add the textual annotation.
    else:
        if complex_anno:
            ion_type = annotation.ion_type[0]
            charge = annotation.charge
            ion_index = annotation.ion_index
        else:
            import re
            tmp_higher = re.compile(r"(y|x|a|b)(\d+)\((\d+)\+\)$")
            tmp = re.compile(r"(y|x|a|b)(\d+)$")
            match = re.match(tmp, annotation)
            matchh = re.match(tmp_higher, annotation)
            if match:
                ion_type = match.group(1)
                ion_index = int(match.group(2))
                charge = 1
            elif matchh:
                ion_type = matchh.group(1)
                ion_index = int(matchh.group(2))
                charge = int(matchh.group(3))
            else:
                print("Error on:", annotation)
                return
        color = (
            COLORS.get(ion_type)
            if color_ions
            else COLORS.get(None)
        )
        zorder = zorders.get(ion_type)
        if annotate_ions:
            annotation_pos = intensity
            anno_mz = mz+5
            if annotation_pos > 0:
                annotation_pos += 0.02
            else:
                annotation_pos -= 0.02
            kws = annotation_kws.copy()
            if abs(annotation_pos) > 0.95:
                kws['rotation'] = 0
            # if overlap:
            #     kws['rotation'] = 90
            del kws["zorder"]
            ax.text(
                anno_mz,
                annotation_pos,
                f"{ion_type}{ion_index}({charge}+)",
                color=color,
                zorder=zorder,
                size=7,
                **kws,
            )

        return color, zorder

def spectrum(
    spec,
    color_ions: bool = True,
    annotate_ions: bool = True,
    annot_kws: Optional[Dict] = None,
    mirror_intensity: bool = False,
    grid: Union[bool, str] = True,
    ax: Optional[plt.Axes] = None,
    colors=COLORS,
    linewidth=3,
    alpha=1,
    label="",
    max_intensity=None
) -> plt.Axes:
    """
    Plot an MS/MS spectrum.
    Parameters
    ----------
    spec : MsmsSpectrum
        The spectrum to be plotted.
    color_ions : bool, optional
        Flag indicating whether or not to color annotated fragment ions. The
        default is True.
    annotate_ions : bool, optional
        Flag indicating whether or not to annotate fragment ions. The default
        is True.
    annot_kws : Optional[Dict], optional
        Keyword arguments for `ax.text` to customize peak annotations.
    mirror_intensity : bool, optional
        Flag indicating whether to flip the intensity axis or not.
    grid : Union[bool, str], optional
        Draw grid lines or not. Either a boolean to enable/disable both major
        and minor grid lines or 'major'/'minor' to enable major or minor grid
        lines respectively.
    ax : Optional[plt.Axes], optional
        Axes instance on which to plot the spectrum. If None the current Axes
        instance is used.
    Returns
    -------
    plt.Axes
        The matplotlib Axes instance on which the spectrum is plotted.
    """
    if ax is None:
        ax = plt.gca()

    min_mz = max(0, math.floor(spec.mz[0] / 100 - 1) * 100)
    max_mz = math.ceil(spec.mz[-1] / 100 + 1) * 100
    ax.set_xlim(min_mz, max_mz)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    y_max = 1.25 if annotate_ions else 1.05
    ax.set_ylim(*(0, y_max) if not mirror_intensity else (-y_max, 0))

    if max_intensity is None:
        max_intensity = spec.intensity.max()
    annotations = (
        spec.annotation
        if spec.annotation is not None
        else itertools.repeat(None)
    )
    annotation_kws = {
        "horizontalalignment": "left" if not mirror_intensity else "right",
        "verticalalignment": "center",
        "rotation": 90,
        "rotation_mode": "anchor",
        "zorder": 5,
    }
    if annot_kws is not None:
        annotation_kws.update(annot_kws)
    for i, (mz, intensity, annotation) in enumerate(zip(spec.mz, spec.intensity, annotations)):
        peak_intensity = intensity / max_intensity
        if mirror_intensity:
            peak_intensity *= -1
        _, zorder = _annotate_ion(
            mz,
            peak_intensity,
            annotation,
            color_ions,
            annotate_ions,
            annotation_kws,
            ax,
        )
        color = colors.get(None)
        if i < (len(spec.mz)-1):
            ax.plot([mz, mz], [0, peak_intensity], color=color, linewidth=linewidth, alpha=alpha, solid_capstyle='round')
        else:
            ax.plot([mz, mz], [0, peak_intensity], color=color, linewidth=linewidth, alpha=alpha, solid_capstyle='round',label=label)

    ax.xaxis.set_minor_locator(mticker.AutoLocator())
    ax.yaxis.set_minor_locator(mticker.AutoLocator())
    ax.xaxis.set_minor_locator(mticker.AutoMinorLocator())
    ax.yaxis.set_minor_locator(mticker.AutoMinorLocator())
    if grid in (True, "both", "major"):
        ax.grid(b=True, which="major", color="#9E9E9E", linewidth=0.2)
    if grid in (True, "both", "minor"):
        ax.grid(b=True, which="minor", color="#9E9E9E", linewidth=0.2)
    ax.set_axisbelow(True)

    ax.tick_params(axis="both", which="both", labelsize="small")
    y_ticks = ax.get_yticks()
    ax.set_yticks(y_ticks[y_ticks <= 1.0])

    ax.set_xlabel("m/z", style="italic", fontsize=12)
    ax.set_ylabel("Intensity", fontsize=12)

    return ax


def mirror(
    spec_top,
    spec_bottom,
    spectrum_kws: Optional[Dict] = None,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """
    Mirror plot two MS/MS spectra.
    Parameters
    ----------
    spec_top : MsmsSpectrum
        The spectrum to be plotted on the top.
    spec_bottom : MsmsSpectrum
        The spectrum to be plotted on the bottom.
    spectrum_kws : Optional[Dict], optional
        Keyword arguments for `plot.spectrum`.
    ax : Optional[plt.Axes], optional
        Axes instance on which to plot the spectrum. If None the current Axes
        instance is used.
    Returns
    -------
    plt.Axes
        The matplotlib Axes instance on which the spectra are plotted.
    """
    if ax is None:
        ax = plt.gca()

    if spectrum_kws is None:
        spectrum_kws = {}
    # Top spectrum.
    spectrum(spec_top, mirror_intensity=False, ax=ax, **spectrum_kws)
    y_max = ax.get_ylim()[1]
    # Mirrored bottom spectrum.
    spectrum(spec_bottom, mirror_intensity=True, ax=ax, **spectrum_kws)
    y_min = ax.get_ylim()[0]
    ax.set_ylim(y_min, y_max)

    ax.axhline(0, color="#9E9E9E", zorder=10)

    # Update axes so that both spectra fit.
    min_mz = max(
        [
            0,
            math.floor(spec_top.mz[0] / 100 - 1) * 100,
            math.floor(spec_bottom.mz[0] / 100 - 1) * 100,
        ]
    )
    max_mz = max(
        [
            math.ceil(spec_top.mz[-1] / 100 + 1) * 100,
            math.ceil(spec_bottom.mz[-1] / 100 + 1) * 100,
        ]
    )
    ax.set_xlim(min_mz, max_mz)
    ax.yaxis.set_major_locator(mticker.AutoLocator())
    ax.yaxis.set_minor_locator(mticker.AutoMinorLocator())
    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, pos: f"{abs(x):.0%}")
    )

    return ax


def spectrum_raw(
    mass, intensities,
    annot_kws: Optional[Dict] = None,
    mirror_intensity: bool = False,
    grid: Union[bool, str] = True,
    ax: Optional[plt.Axes] = None,
    specical_peak=None,
    colors=COLORS,
    linewidth=3,
    alpha=1,
    label=""
) -> plt.Axes:
    if ax is None:
        ax = plt.gca()
    mass = np.array(mass)
    intensities = np.array(intensities)
    min_mz = 0
    max_mz = math.ceil(mass.max() / 100 + 1) * 100
    ax.set_xlim(min_mz, max_mz)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    y_max = 1.05
    ax.set_ylim(*(0, y_max) if not mirror_intensity else (-y_max, 0))

    max_intensity = intensities.max()
    for i, (mz, intensity) in enumerate(zip(mass, intensities)):
        peak_intensity = intensity / max_intensity
        if mirror_intensity:
            peak_intensity *= -1
        color = colors.get(None)
        if specical_peak is not None:
            for sp_mz in specical_peak:
                if abs(mz - sp_mz) < 1e-7:
                    color = colors.get("special", "cyan")
                    break
        if i < (len(mass)-1):
            ax.plot([mz, mz], [-0.01, peak_intensity], color=color, 
                    linewidth=linewidth, alpha=alpha, solid_capstyle='round')
        else:
            ax.plot([mz, mz], [-0.01, peak_intensity], color=color, 
                    linewidth=linewidth, alpha=alpha, solid_capstyle='round', label=label)

    ax.xaxis.set_minor_locator(mticker.AutoLocator())
    ax.yaxis.set_minor_locator(mticker.AutoLocator())
    ax.xaxis.set_minor_locator(mticker.AutoMinorLocator())
    ax.yaxis.set_minor_locator(mticker.AutoMinorLocator())
    if grid in (True, "both", "major"):
        ax.grid(b=True, which="major", color="#9E9E9E", linewidth=0.2)
    if grid in (True, "both", "minor"):
        ax.grid(b=True, which="minor", color="#9E9E9E", linewidth=0.2)
    ax.set_axisbelow(True)

    ax.tick_params(axis="both", which="both", labelsize="small")
    y_ticks = ax.get_yticks()
    ax.set_yticks(y_ticks[y_ticks <= 1.0])

    ax.set_xlabel("m/z", style="italic", fontsize=12)
    ax.set_ylabel("Intensity", fontsize=12)

    return ax


def spectrum_ions_offered(
    ions, intensities, masses, 
    color_ions: bool = True,
    annotate_ions: bool = True,
    annot_kws: Optional[Dict] = None,
    mirror_intensity: bool = False,
    grid: Union[bool, str] = True,
    ax: Optional[plt.Axes] = None,
    colors=COLORS,
    linewidth=3,
    alpha=1,
    label="",
    max_intensity=None,
    min_intensity=0.05,
    max_mz = None
) -> plt.Axes:
    """
    Plot an MS/MS spectrum.
    Parameters
    ----------
    spec : MsmsSpectrum
        The spectrum to be plotted.
    color_ions : bool, optional
        Flag indicating whether or not to color annotated fragment ions. The
        default is True.
    annotate_ions : bool, optional
        Flag indicating whether or not to annotate fragment ions. The default
        is True.
    annot_kws : Optional[Dict], optional
        Keyword arguments for `ax.text` to customize peak annotations.
    mirror_intensity : bool, optional
        Flag indicating whether to flip the intensity axis or not.
    grid : Union[bool, str], optional
        Draw grid lines or not. Either a boolean to enable/disable both major
        and minor grid lines or 'major'/'minor' to enable major or minor grid
        lines respectively.
    ax : Optional[plt.Axes], optional
        Axes instance on which to plot the spectrum. If None the current Axes
        instance is used.
    Returns
    -------
    plt.Axes
        The matplotlib Axes instance on which the spectrum is plotted.
    """
    if ax is None:
        ax = plt.gca()
    mass_order = np.argsort(masses)
    masses = np.array(masses)[mass_order]
    intensities = np.array(intensities)[mass_order]
    ions = [ions[i] for i in mass_order]
    min_mz = max(0, math.floor(masses.min() / 100 - 1) * 100)
    if max_mz is None:
        max_mz = math.ceil(masses.max() / 100 + 1) * 100
    ax.set_xlim(min_mz, max_mz)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    y_max = 1.25 if annotate_ions else 1.05
    ax.set_ylim(*(0, y_max) if not mirror_intensity else (-y_max, 0))

    if max_intensity is None:
        max_intensity = intensities.max()
    annotation_kws = {
        "horizontalalignment": "left" if not mirror_intensity else "right",
        "verticalalignment": "center",
        "rotation": 90,
        "rotation_mode": "anchor",
        "zorder": 5,
    }
    if annot_kws is not None:
        annotation_kws.update(annot_kws)
    for i, (mz, intensity, annotation) in enumerate(zip(masses, intensities, ions)):
        peak_intensity = intensity / max_intensity
        if peak_intensity < min_intensity or mz > max_mz:
            continue

        if mirror_intensity:
            peak_intensity *= -1
        _, zorder = _annotate_ion(
            mz,
            peak_intensity,
            annotation,
            color_ions,
            annotate_ions,
            annotation_kws,
            ax,
            complex_anno=False,
            overlap=(i != 0 and (mz - masses[i-1]) < 20)
        )
        color = colors.get(None)
        if i < (len(masses)-1):
            ax.plot([mz, mz], [0, peak_intensity], color=color,
                    linewidth=linewidth, alpha=alpha, solid_capstyle='round')
        else:
            ax.plot([mz, mz], [0, peak_intensity], color=color, linewidth=linewidth,
                    alpha=alpha, solid_capstyle='round', label=label)

    ax.xaxis.set_minor_locator(mticker.AutoLocator())
    ax.yaxis.set_minor_locator(mticker.AutoLocator())
    ax.xaxis.set_minor_locator(mticker.AutoMinorLocator())
    ax.yaxis.set_minor_locator(mticker.AutoMinorLocator())
    if grid in (True, "both", "major"):
        ax.grid(b=True, which="major", color="#9E9E9E", linewidth=0.2)
    if grid in (True, "both", "minor"):
        ax.grid(b=True, which="minor", color="#9E9E9E", linewidth=0.2)
    ax.set_axisbelow(True)

    ax.tick_params(axis="both", which="both", labelsize="small")
    y_ticks = ax.get_yticks()
    ax.set_yticks(y_ticks[y_ticks <= 1.0])

    ax.set_xlabel("m/z", style="italic", fontsize=12)
    ax.set_ylabel("Intensity", fontsize=12)

    return ax
