# -*- coding: utf-8 -*-
from typing import Dict
from typing import List
from typing import Union

import numpy as np
from mrsimulator import Site
from mrsimulator import SpinSystem
from mrsimulator.base_model import populate_sites
from mrsimulator.base_model import populate_spin_systems

__author__ = ["Deepansh Srivastava", "Matthew D. Giammar"]
__email__ = ["srivastava.89@osu.edu", "giammar.7@buckeyemail.osu.edu"]

SHIELDING_SYM_PARAMS = ["zeta", "eta", "alpha", "beta", "gamma"]
QUADRUPOLAR_PARAMS = ["Cq", "eta", "alpha", "beta", "gamma"]
LIST_LEN_ERROR_MSG = (
    "All arguments must be the same size. If one attribute is a type list of length n, "
    "then all attributes with list types must also be of length n, and all remaining "
    "attributes must be scalar (singular float, int, or str)."
)


def single_site_system_generator(
    isotope: Union[str, List[str]],
    isotropic_chemical_shift: Union[float, List[float], np.ndarray] = 0,
    shielding_symmetric: Dict = None,
    shielding_antisymmetric: Dict = None,
    quadrupolar: Dict = None,
    abundance: Union[float, List[float], np.ndarray] = None,
    site_name: Union[str, List[str]] = None,
    site_label: Union[str, List[str]] = None,
    site_description: Union[str, List[str]] = None,
    rtol: float = 1e-3,
) -> List[SpinSystem]:
    r"""Generate and return a list of single-site spin systems from the input parameters.

    Args:
        isotope:
            A required string or a list of site isotopes.
        isotropic_chemical_shift:
            A float or a list/ndarray of isotropic chemical shifts per site per spin
            system. The default is 0.
        shielding_symmetric:
            A shielding symmetric dict object, where the keyword value can either
            be a float or a list/ndarray of floats. The default value is None. The
            allowed keywords are ``zeta``, ``eta``, ``alpha``, ``beta``, and ``gamma``.
        shielding_antisymmetric:
            A shielding antisymmetric dict object, where the keyword value can either
            be a float or a list/ndarray of floats. The default value is None. The
            allowed keywords are ``zeta``, ``alpha``, and ``beta``.
        quadrupolar:
            A quadrupolar dict object, where the keyword value can either be a float or
            a list/ndarray of floats. The default value is None. The allowed keywords
            are ``Cq``, ``eta``, ``alpha``, ``beta``, and ``gamma``.
        abundance:
            A float or a list/ndarray of floats describing the abundance of each spin
            system.
        site_name:
            A string or a list of strings with site names per site per spin system. The
            default is None.
        site_label:
            A string or a list of strings with site labels per site per spin system. The
            default is None.
        site_description:
            A string or a list of strings with site descriptions per site per spin
            system. The default is None.
        rtol:
            The relative tolerance used in determining the cutoff abundance, given as,
            :math:`\tt{abundance}_{\tt{cutoff}} = \tt{rtol} * \tt{max(abundance)}.`
            The spin systems with abundance below this threshold are ignored.

    Returns:
        List of :ref:`spin_sys_api` objects with a single :ref:`site_api`

    Example:
        **Single spin system:**

        >>> sys1 = single_site_system_generator(
        ...     isotope=["1H"],
        ...     isotropic_chemical_shift=10,
        ...     site_name="Single Proton",
        ... )
        >>> print(len(sys1))
        1

        **Multiple spin system:**

        >>> sys2 = single_site_system_generator(
        ...     isotope="1H",
        ...     isotropic_chemical_shift=[10] * 5,
        ...     site_name="5 Protons",
        ... )
        >>> print(len(sys2))
        5

        **Multiple spin system with dictionary arguments:**

        >>> Cq = [4.2e6] * 12
        >>> sys3 = single_site_system_generator(
        ...     isotope="17O",
        ...     isotropic_chemical_shift=60.0,  # in ppm,
        ...     quadrupolar={"Cq": Cq, "eta": 0.5},  # Cq in Hz
        ... )
        >>> print(len(sys3))
        12

    .. note::
        The parameter value can either be a float or a list/ndarray. If the parameter
        value is a float, the given value is assigned to the respective parameter in all
        the spin systems. If the parameter value is a list or ndarray, its `ith` value
        is assigned to the respective parameter of the `ith` spin system. When multiple
        parameter values are given as lists/ndarrays, the length of all the lists must
        be the same.
    """
    sites = site_generator(
        isotope=isotope,
        isotropic_chemical_shift=isotropic_chemical_shift,
        shielding_symmetric=shielding_symmetric,
        shielding_antisymmetric=shielding_antisymmetric,
        quadrupolar=quadrupolar,
        name=site_name,
        label=site_label,
        description=site_description,
    )
    n_sites = len(sites)

    # Compute abundances
    abundance = 1 / n_sites if abundance is None else abundance
    abundance = _extend_to_nparray(_fix_item(abundance), n_sites)
    n_abd = abundance.size

    # Extend sites based on abundance if number of sites is 1
    if n_sites == 1:
        sites = np.asarray([sites[0] for _ in range(n_abd)])
        n_sites = sites.size

    # List mismatch for sites and abundances
    if n_sites != n_abd:
        raise ValueError(
            "Number of sites does not match the number of abundances. "
            f"{LIST_LEN_ERROR_MSG}"
        )

    # Calculate kept sites based on abundance tolerance
    keep_idxs = np.where(abundance > rtol * abundance.max())[0]
    sites = np.asarray(sites)[keep_idxs]
    abundance = abundance[keep_idxs]
    n_sys = keep_idxs.size

    return populate_spin_systems(
        spin_systems=[SpinSystem() for _ in range(n_sys)],
        sites=sites,
        abundance=abundance,
    )


def site_generator(
    isotope: Union[str, List[str]],
    isotropic_chemical_shift: Union[float, List[float], np.ndarray] = 0,
    shielding_symmetric: Dict = None,
    shielding_antisymmetric: Dict = None,
    quadrupolar: Dict = None,
    name: Union[str, List[str]] = None,
    label: Union[str, List[str]] = None,
    description: Union[str, List[str]] = None,
) -> List[Site]:
    r"""Generate a list of Site objects from lists of site attributes.

    Args:
        isotope:
            A required string or a list of site isotopes.
        isotropic_chemical_shift:
            A float or a list/ndarray of isotropic chemical shifts per site. The default
            is 0.
        shielding_symmetric:
            A shielding symmetric dict object, where the keyword value can either
            be a float or a list/ndarray of floats. The default value is None. The
            allowed keywords are ``zeta``, ``eta``, ``alpha``, ``beta``, and ``gamma``.
        shielding_antisymmetric:
            A shielding antisymmetric dict object, where the keyword value can either
            be a float or a list/ndarray of floats. The default value is None. The
            allowed keywords are ``zeta``, ``alpha``, and ``beta``.
        quadrupolar:
            A quadrupolar dict object, where the keyword value can either be a float or
            a list/ndarray of floats. The default value is None. The allowed keywords
            are ``Cq``, ``eta``, ``alpha``, ``beta``, and ``gamma``.
        name:
            A string or a list of strings with site names per site. The default is None.
        label:
            A string or a list of strings with site labels per site. The default is
            None.
        description:
            A string or a list of strings with site descriptions per site. The default
            is None.

    Returns:
        sites: List of :ref:`site_api` objects

    Example:
        **Generating 10 hydrogen sites:**

        >>> sites1 = site_generator(
        ...     isotope=["1H"] * 10,
        ...     isotropic_chemical_shift=-15,
        ...     name="10 Protons",
        ... )
        >>> print(len(sites1))
        10

        **Generating 10 hydrogen sites with different shifts:**

        >>> shifts = np.arange(-10, 10, 2)
        >>> sites2 = site_generator(
        ...     isotope=["1H"] * 10,
        ...     isotropic_chemical_shift=shifts,
        ...     name="10 Proton",
        ... )
        >>> print(len(sites2))
        10

        **Generating multiple sites with dictionary arguments:**

        >>> Cq = [4.2e6] * 12
        >>> sys3 = site_generator(
        ...     isotope="17O",
        ...     isotropic_chemical_shift=60.0,  # in ppm,
        ...     quadrupolar={"Cq": Cq, "eta": 0.5},  # Cq in Hz
        ... )
        >>> print(len(sys3))
        12
    """
    attr_kw = {
        "isotope": _fix_item(isotope),
        "isotropic_chemical_shift": _fix_item(isotropic_chemical_shift),
        "name": _fix_item(name),
        "label": _fix_item(label),
        "description": _fix_item(description),
    }

    # Add dict vals to attr_kw with key 'prefix_' + 'key'
    _add_dict_to_attr(
        attr_kw=attr_kw,
        _dict=shielding_symmetric,
        prefix="shielding_symmetric",
        keys=["zeta", "eta", "alpha", "beta", "gamma"],
    )
    _add_dict_to_attr(
        attr_kw=attr_kw,
        _dict=shielding_antisymmetric,
        prefix="shielding_antisymmetric",
        keys=["zeta", "alpha", "beta"],
    )
    _add_dict_to_attr(
        attr_kw=attr_kw,
        _dict=quadrupolar,
        prefix="quadrupolar",
        keys=["Cq", "eta", "alpha", "gamma"],
    )

    # Find number of sites and extend non-array values to np.array of length n_sites
    n_sites = _check_lengths(attr_kw.values())
    attr_kw = {key: _extend_to_nparray(val, n_sites) for key, val in attr_kw.items()}

    # NOTE: Thinking about using zip() and pass alternate array to c function
    return populate_sites(
        sites=[Site() for _ in range(n_sites)],
        **attr_kw,
    )


def _fix_item(item):
    """Flattens multidimensional arrays into 1d array."""
    if isinstance(item, (list, np.ndarray)):
        return np.asarray(item).ravel()
    return item


def _extend_to_nparray(item, n):
    """If item is already list/array return np.array, otherwise extend to length n."""
    data = item if isinstance(item, (list, np.ndarray)) else [item for _ in range(n)]
    return np.asarray(data)


def _add_dict_to_attr(attr_kw, _dict, prefix, keys):
    """Ensures all keys are present in _dict, if not adds 'key': None. Then adds
    keys into attr_kw"""
    if _dict is None:
        _dict = {key: None for key in keys}
    else:
        _dict = {key: (_fix_item(_dict[key]) if key in _dict else None) for key in keys}
    attr_kw.update([(f"{prefix}_{key}", _dict[key]) for key in keys])


def _check_lengths(attributes):
    """Ensures all attribute lengths are 1 or maximum attribute length."""
    lengths = np.array([np.asarray(attr).size for attr in attributes])

    if np.all(lengths == 1):
        return 1

    lengths = lengths[np.where(lengths != 1)]
    if np.unique(lengths).size == 1:
        return lengths[0]

    raise ValueError(
        f"An array or list was either too short or too long. {LIST_LEN_ERROR_MSG}"
    )
