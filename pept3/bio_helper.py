import enum
import numpy
import numpy as np
from torch._C import dtype
from .constants import AMINO_ACID, PROTON, ION_OFFSET, FORWARD, BACKWARD
from . import constants
import collections


def peptide_parser(p):
    p = p.replace("_", "")
    p = p.replace(".", "")
    if p[0] == "(":
        print(p)
        raise ValueError("sequence starts with '('")
    n = len(p)
    i = 0
    while i < n:
        if i < n - 3 and p[i + 1] == "(":
            j = p[i + 2:].index(")")
            offset = i + j + 3
            yield p[i:offset]
            i = offset
        else:
            yield p[i]
            i += 1


def peptide_to_inter(seq, max_length=30):
    re = np.zeros((max_length, ), dtype='int')
    for i, s in enumerate(peptide_parser(seq)):
        if len(s) > 1:
            s = s[0].upper()+s[1:]
        else:
            s = s.upper()
        re[i] = constants.ALPHABET[s]
    return re.reshape(1, -1)


def one_hot(flag, max_cate=6):
    re = np.zeros((max_cate, ))
    re[flag] = 1
    return re.reshape(1, -1)


def adjust_masses(method):
    if method == "SILAC":
        offsets = {"K": 8.01419881319, "R": 10.008268599}
    else:
        raise ValueError("Don't know method: " + method)

    for aa, offset in offsets.items():
        AMINO_ACID[aa] += offset


def get_mz(sum_, ion_offset, charge):
    return (sum_ + ion_offset + charge * PROTON) / charge


def get_mzs(cumsum, ion_type, z):
    return [get_mz(s, ION_OFFSET[ion_type], z) for s in cumsum[:-1]]


def get_annotation(forward, backward, charge, ion_types, offset=0):
    tmp = "{}{}"
    tmp_higher = "{}{}"
    tmp_nl = "{}{}-{}"
    all_ = {}
    for ion_type in ion_types:
        if ion_type in constants.FORWARD:
            cummass = forward
        elif ion_type in constants.BACKWARD:
            cummass = backward
        else:
            raise ValueError("unkown ion_type: {}".format(ion_type))
        masses = get_mzs(cummass, ion_type, charge)
        d = {tmp.format(ion_type, i + 1): m for i, m in enumerate(masses)}
        all_.update(d)
    return collections.OrderedDict(sorted(all_.items(), key=lambda t: t[0]))


def reverse_annotation(matches, intensities, charges, length):
    import re
    tmp = re.compile(r"(y|x|a|b)(\d+)$")
    tmp_nl = re.compile(r"(y|x|a|b)(\d+)-(NH3|H2O)$")
    tmp_higher = re.compile(r"(y|x|a|b)(\d+)\((\d+)\+\)$")
    tmp_higher_nl = re.compile(r"(y|x|a|b)(\d+)-(NH3|H2O)\((\d+)\+\)$")
    result = np.zeros((29, 2, 3))
    ion_dict = {
        'y': 0, 'b': 1
    }
    for m, inten in zip(matches, intensities):
        if re.match(tmp_nl, m) or re.match(tmp_higher_nl, m):
            # No Natural ions considered
            continue
        match = re.match(tmp, m)
        matchh = re.match(tmp_higher, m)
        if match:
            ion = match.group(1)
            frag_i = int(match.group(2))
            charge = 1
        elif matchh:
            ion = matchh.group(1)
            frag_i = int(matchh.group(2))
            charge = int(matchh.group(3))
        else:
            raise TypeError(f"{m} can't be parsed")
        if ion not in ion_dict or charge > 3:
            continue
        result[frag_i - 1, ion_dict[ion], charge - 1] = float(inten)
    result[:, :, charges:] = -1
    result[length - 1:] = -1
    return result


def read_attribute(row, attribute):
    if " " not in str(row[attribute]):
        return []
    else:
        return [float(m) for m in row[attribute].split(" ")]


# def peptide_parser(p):
#     if p[0] == "(":
#         raise ValueError("sequence starts with '('")
#     n = len(p)
#     i = 0
#     while i < n:
#         if i < n - 3 and p[i + 1] == "(":
#             j = p[i + 2:].index(")")
#             offset = i + j + 3
#             yield p[i:offset]
#             i = offset
#         else:
#             yield p[i]
#             i += 1


def get_forward_backward(peptide, offset=0):
    amino_acids = peptide_parser(peptide)
    masses = [constants.AMINO_ACID[a] for a in amino_acids]
    forward = numpy.cumsum(masses) - offset
    backward = numpy.cumsum(list(reversed(masses))) - offset
    return forward, backward


def get_tolerance(theoretical, mass_analyzer):
    if mass_analyzer in constants.TOLERANCE:
        tolerance, unit = constants.TOLERANCE[mass_analyzer]
        if unit == "ppm":
            return theoretical * float(tolerance) / 10 ** 6
        elif unit == "da":
            return float(tolerance)
        else:
            raise ValueError("unit {} not implemented".format(unit))
    else:
        raise ValueError(
            "no tolerance implemented for {}".format(mass_analyzer))


def is_in_tolerance(theoretical, observed, mass_analyzer):
    mz_tolerance = get_tolerance(theoretical, mass_analyzer)
    lower = observed - mz_tolerance
    upper = observed + mz_tolerance
    return theoretical >= lower and theoretical <= upper


def binarysearch(masses_raw, theoretical, mass_analyzer):
    lo, hi = 0, len(masses_raw) - 1
    masses_raw = sorted(masses_raw)
    while lo <= hi:
        mid = (lo + hi) // 2
        if is_in_tolerance(theoretical, masses_raw[mid], mass_analyzer):
            return mid
        elif masses_raw[mid] < theoretical:
            lo = mid + 1
        elif theoretical < masses_raw[mid]:
            hi = mid - 1
    return None


def pair_backbone_with_mass(pred_inten, peptide, charge):
    import re
    tmp = re.compile(r"(y|x|a|b)(\d+)$")

    forward_sum, backward_sum = get_forward_backward(peptide)
    pred_inten = pred_inten.reshape(29, 2, 3)
    ion_dict = {
        'y': 0, 'b': 1
    }
    # result[frag_i-1, ion_dict[ion], charge-1] = float(inten)
    intens = []
    masses = []
    annos = []
    for c_index in range(min(charge, 3)):
        annotations = get_annotation(
            forward_sum, backward_sum, c_index + 1, "by"
        )
        for anno, mass_t in annotations.items():
            match = re.match(tmp, anno)
            if match:
                annos.append(anno)
                ion = match.group(1)
                frag_i = int(match.group(2))
                # print(frag_i - 1, ion_dict[ion], c_index)
                intens.append(pred_inten[frag_i - 1, ion_dict[ion], c_index])
                masses.append(mass_t)
    return intens, masses


def match(row, ion_types, max_charge=constants.DEFAULT_MAX_CHARGE):
    masses_observed = read_attribute(row, "masses_raw")
    intensities_observed = read_attribute(row, "intensities_raw")
    forward_sum, backward_sum = get_forward_backward(
        row['modified_sequence'])
    _max_charge = row['charge'] if row['charge'] <= max_charge else max_charge
    matches = {}
    for charge_index in range(_max_charge):
        d = {
            "masses_raw": [],
            "masses_theoretical": [],
            "intensities_raw": [],
            "matches": [],
        }
        charge = charge_index + 1
        annotations = get_annotation(
            forward_sum, backward_sum, charge, ion_types
        )
        for annotation, mass_t in annotations.items():
            index = binarysearch(masses_observed, mass_t, row['mass_analyzer'])
            if index is not None:
                d["masses_raw"].append(masses_observed[index])
                d["intensities_raw"].append(intensities_observed[index])
                d["masses_theoretical"].append(mass_t)
                d["matches"].append(annotation)
        matches[charge] = d
    return matches


def match_all(row, ion_types='yb', max_charge=constants.DEFAULT_MAX_CHARGE):
    matches = match(row, ion_types, max_charge)
    intensities_observed = read_attribute(row, "intensities_raw")
    total_peaks = len(intensities_observed)
    scale_inten = np.sqrt(np.sum(np.array(intensities_observed)**2))
    tmp = "{}({}+)"
    all_ions = []
    all_intensities = []
    all_masses = []
    all_charge = row['charge'] if row['charge'] <= max_charge else max_charge
    peptides_leng = len(list(peptide_parser(row['modified_sequence'])))
    for charge, ions in matches.items():
        all_intensities.extend(ions['intensities_raw'])
        all_masses.extend(ions['masses_raw'])
        if charge == 1:
            all_ions.extend(ions['matches'])
        else:
            match_ions = [tmp.format(i, str(charge)) for i in ions['matches']]
            all_ions.extend(match_ions)
    return all_ions, all_intensities, all_charge, peptides_leng, len(all_intensities) / (total_peaks + 1e-9), all_masses, scale_inten


def match_all_multi(row, ion_types='yb', max_charge=constants.DEFAULT_MAX_CHARGE):
    matches = match(row, ion_types, max_charge)
    intensities_observed = read_attribute(row, "intensities_raw")
    total_peaks = len(intensities_observed)
    scale_inten = np.sqrt(np.sum(np.array(intensities_observed)**2))
    tmp = "{}({}+)"
    all_ions = []
    all_intensities = []
    all_masses = []
    all_charge = row['charge'] if row['charge'] <= max_charge else max_charge
    peptides_leng = len(list(peptide_parser(row['modified_sequence'])))
    for charge, ions in matches.items():
        all_intensities.extend(ions['intensities_raw'])
        all_masses.extend(ions['masses_raw'])
        if charge == 1:
            all_ions.extend(ions['matches'])
        else:
            match_ions = [tmp.format(i, str(charge)) for i in ions['matches']]
            all_ions.extend(match_ions)
    return row['id'], all_ions, all_intensities, all_charge, peptides_leng, len(all_intensities) / (total_peaks + 1e-9), all_masses, scale_inten


def c_lambda(matches, charge, attr):
    def mapping(i):
        charge_index = int(charge - 1)
        m = matches[i]
        if charge_index < len(m):
            try:
                s = ";".join(map(str, m[charge_index][attr]))
            except:
                raise ValueError(m[charge_index][attr])
        else:
            s = ""
        return s

    return mapping


def augment(df, ion_types, charge_max):
    matches = {}
    for i, row in df.iterrows():
        matches[i] = match(row, ion_types, charge_max)

    # augment dataframe and write
    for charge in range(1, charge_max + 1):
        df["matches_charge{}".format(charge)] = df.index.map(
            c_lambda(matches, charge, "matches")
        )
        df["masses_the_charge{}".format(charge)] = df.index.map(
            c_lambda(matches, charge, "masses_theoretical")
        )
        df["masses_raw_charge{}".format(charge)] = df.index.map(
            c_lambda(matches, charge, "masses_raw")
        )
        df["intensities_raw_charge{}".format(charge)] = df.index.map(
            c_lambda(matches, charge, "intensities_raw")
        )

    return df
