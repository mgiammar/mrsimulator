# -*- coding: utf-8 -*-
from typing import List

import numpy as np
from mrsimulator import Site
from mrsimulator import SpinSystem

__author__ = ["Deepansh Srivastava", "Matthew D. Giammar"]
__email__ = ["srivastava.89@osu.edu", "giammar.7@buckeyemail.osu.edu"]

SHIELDING_SYM_PARAMS = ["zeta", "eta", "alpha", "beta", "gamma"]
QUADRUPOLAR_PARAMS = ["Cq", "eta", "alpha", "beta", "gamma"]
LIST_LEN_ERROR_MSG = (
    "All arguments passed must be the same size. If one attribute is type list of "
    "length n, then all passed attributes must be type list of length n or scalar. "
    "Scalar (singular float, int or str) attributes will be extended to the length of "
    "isotopes"
)


# NOTE: Should args all be plural or all be singular


def single_site_system_generator(
    isotopes,
    isotropic_chemical_shifts=0,
    shielding_symmetric=None,
    shielding_antisymmetric=None,
    quadrupolars=None,
    abundances=None,
    site_names=None,
    site_labels=None,
    site_descriptions=None,
    rtol=1e-3,
):
    r"""Generate and return a list of single-site spin systems from the input parameters.

    Args:

        (list) isotopes:
            A required string or a list of site isotopes.
        (list) isotropic_chemical_shifts:
            A float or a list/ndarray of values. The default value is 0.
        (list) shielding_symmetric:
            A shielding symmetric dict-like object, where the keyword value can either
            be a float or a list/ndarray of floats. The default value is None. The
            allowed keywords are ``zeta``, ``eta``, ``alpha``, ``beta``, and ``gamma``.
        (dict) sheilding_antisymmetric:
            A shielding symmetric dict-like object, where the keyword value can either
            be a float or a list/ndarray of floats. The default value is None. The
            allowed keywords are ``zeta``, ``alpha``, and ``beta``.
        (dict) quadrupolars:
            A quadrupolar dict-like object, where the keyword value can either be a
            float or a list/ndarray of floats. The default value is None. The allowed
            keywords are ``Cq``, ``eta``, ``alpha``, ``beta``, and ``gamma``.
        (list) abundances:
            A float or a list/ndarray of floats describing the abundances of each spin
            system.
        (list) site_names:
            A string or a list of strings each with a site name. The default is None.
        (list) site_labels:
            A string or a list of strings each with a site label. The default is None.
        (list) site_descriptions:
            A string or a list of strings each with a site description. Default is None.
        (float) rtol:
            The relative tolerance. This value is used in determining the cutoff
            abundances given as
            :math:`\tt{abundances}_{\tt{cutoff}} = \tt{rtol} * \tt{max(abundances)}.`
            The spin systems with abundances below this threshold are ignored.

    Returns:
        TODO add return type

    Example:
        TODO add example code

    .. note::
        The parameter value can either be a float or a list/ndarray. If the parameter
        value is a float, the given value is assigned to the respective parameter in all
        the spin systems. If the parameter value is a list or ndarray, its ith value is
        assigned to the respective parameter of the ith spin system. When multiple
        parameter values are given as lists/ndarrays, the length of all the lists must
        be the same.
    """
    sites = generate_site_list(
        isotopes=isotopes,
        isotropic_chemical_shifts=isotropic_chemical_shifts,
        shielding_symmetric=shielding_symmetric,
        shielding_antisymmetric=shielding_antisymmetric,
        quadrupolars=quadrupolars,
        site_names=site_names,
        site_labels=site_labels,
        site_descriptions=site_descriptions,
    )
    n_sites = len(sites)

    if abundances is None:
        abundances = 1 / n_sites
    abundances = _extend_to_nparray(abundances, n_sites)
    n_abd = abundances.size

    if n_sites == 1:
        sites = sites * n_abd
        n_sites = len(sites)

    if n_sites != n_abd:
        raise ValueError(
            "Number of sites does not mach number of abundancess. " + LIST_LEN_ERROR_MSG
        )

    keep_idxs = np.where(abundances > rtol * abundances.max())[0]

    return [
        SpinSystem(sites=[site], abundances=abd)
        for site, abd in zip(sites[keep_idxs], abundances[keep_idxs])
    ]


def generate_site_list(
    isotopes,
    isotropic_chemical_shifts=0,
    shielding_symmetric=None,
    shielding_antisymmetric=None,
    quadrupolars=None,
    site_names=None,
    site_labels=None,
    site_descriptions=None,
) -> List[Site]:
    r"""Takes in lists or list-like objects describing attributes of each site and
    returns a list of Site objects (TODO add doc ref)

    Params:
        (list) isotopes:
            A string or a list of site isotopes.
        (list) isotropic_chemical_shifts:
            A float or a list/ndarray of values. The default value is 0.
        (dict) shielding_symmetric:
            A shielding symmetric dict-like object, where the keyword value can either
            be a float or a list/ndarray of floats. The default value is None. The
            allowed keywords are ``zeta``, ``eta``, ``alpha``, ``beta``, and ``gamma``.
        (dict) sheilding_antisymmetric:
            A shielding symmetric dict-like object, where the keyword value can either
            be a float or a list/ndarray of floats. The default value is None. The
            allowed keywords are ``zeta``, ``alpha``, and ``beta``.
        (dict) quadrupolars:
            A quadrupolar dict-like object, where the keyword value can either be a
            float or a list/ndarray of floats. The default value is None. The allowed
            keywords are ``Cq``, ``eta``, ``alpha``, ``beta``, and ``gamma``.
        (list) site_names:
            A string or a list of strings each with a site name. The default is None.
        (list) site_labels:
            A string or a list of strings each with a site label. The default is None.
        (list) site_descriptions:
            A string or a list of strings each with a site description. Default is None.

    Returns:
        (list) sites: List of Site objects (TODO add doc ref)

    Example:
        TODO add example code
    """
    attributes = [
        isotopes,
        isotropic_chemical_shifts,
        site_names,
        site_labels,
        site_descriptions,
    ]

    # NOTE: Cannot guess number of sites from dictonary parameters
    n_sites = _check_lengths(attributes)
    attributes = [_extend_to_nparray(attr, n_sites) for attr in attributes]

    if shielding_symmetric is not None:
        attributes.append(_zip_dict(_extend_dict_values(shielding_symmetric, n_sites)))
    else:
        attributes.append(_extend_to_nparray(None, n_sites))

    if shielding_antisymmetric is not None:
        attributes.append(
            _zip_dict(_extend_dict_values(shielding_antisymmetric, n_sites))
        )
    else:
        attributes.append(_extend_to_nparray(None, n_sites))

    if quadrupolars is not None:
        attributes.append(_zip_dict(_extend_dict_values(quadrupolars, n_sites)))
    else:
        attributes.append(_extend_to_nparray(None, n_sites))

    return np.asarray(
        [
            Site(
                isotope=iso,
                isotropic_chemical_shift=shift,
                shielding_symmetric=symm,
                shielding_antisymmetric=antisymm,
                quadrupolar=quad,
                name=name,
                label=label,
                description=desc,
            )
            for iso, shift, symm, antisymm, quad, name, label, desc in zip(*attributes)
        ]
    )


def _zip_dict(_dict):
    """Makes list of dicts with the same keys and scalar values from dict of lists

    Example:
    >>> foo = {'key1': [1, 2, 3, 4], 'key2': [5, 6, 7, 8], 'key3': [9, 10, 11, 12]}
    >>> _zip_dict(foo)
    [
        {'key1': 1, 'key2': 5, 'key3': 9},
        {'key1': 2, 'key2': 6, 'key3': 10},
        {'key1': 3, 'key2': 7, 'key3': 11},
        {'key1': 4, 'key2': 8, 'key3': 12},
    ]
    """
    return [dict(zip(_dict.keys(), v)) for v in zip(*(_dict[k] for k in _dict.keys()))]


def _extend_to_nparray(item, n):
    """If item is already list/array return np.array, otherwise extend to length n"""
    if isinstance(item, (list, np.ndarray)):
        return np.asarray(item)
    return np.asarray([item for _ in range(n)])


def _extend_dict_values(_dict, n):
    """Extends scalar dict values to np.array of length n and checks lengths"""
    _dict = {key: _extend_to_nparray(val, n) for key, val in _dict.items()}

    lengths = np.array([val.size for val in _dict.values()])
    if np.any(lengths != n):
        raise ValueError("A list in a dictonary was misshapen. " + LIST_LEN_ERROR_MSG)

    return _dict


def _check_lengths(attributes):
    """Ensures all attribute lengths are 1 or maximum attribute length"""
    lengths = np.array([np.asarray(attr).size for attr in attributes])

    if np.all(lengths == 1):
        return 1

    lengths = lengths[np.where(lengths != 1)]
    if np.unique(lengths).size == 1:
        return lengths[0]

    raise ValueError(
        "An array or list was either too short or too long. " + LIST_LEN_ERROR_MSG
    )


# def _check_input_list_lengths(attributes):
#     """Ensures all input list lengths are the same"""
#     lengths = np.asarray(
#         [
#             attr.size if isinstance(attr, np.ndarray) else list(attr.values())[0].size
#             for attr in attributes
#         ]
#     )
#     if np.any(lengths != lengths[0]):
#         bad_list = attributes[np.where(lengths != lengths[0])[0][0]]
#         good_len = attributes[0].size
#         raise ValueError(
#             "An array or list was either too short or too long. "
#             + LIST_LEN_ERROR_MSG
#             + f"{bad_list} is size ({len(bad_list)}) should be size ({good_len})"
#         )
#     return


# def _clean_item(item, n):
#     """Cleanes passed item to np.array"""
#     # Return flattened np.array if item is already list or array
#     if isinstance(item, (list, np.ndarray)):
#         return np.hstack(np.asarray(item, dtype=object))
#     # Return default value extended to specified length
#     return np.asarray([item for _ in range(n)])


# def _get_shielding_info(shielding_symmetric):
#     n_ss, shield_keys = [], []
#     if shielding_symmetric is not None:
#         shield_keys = shielding_symmetric.keys()
#         shielding_symmetric = {
#             item: _flatten_item(shielding_symmetric[item])
#             for item in SHIELDING_SYM_PARAMS
#             if item in shield_keys
#         }
#         n_ss = [
#             _get_length(shielding_symmetric[item])
#             for item in SHIELDING_SYM_PARAMS
#             if item in shield_keys
#         ]
#     return n_ss, shield_keys, shielding_symmetric


# def _get_quad_info(quadrupolar):
#     n_q, quad_keys = [], []
#     if quadrupolar is not None:
#         quad_keys = quadrupolar.keys()
#         quadrupolar = {
#             item: _flatten_item(quadrupolar[item])
#             for item in QUADRUPOLAR_PARAMS
#             if item in quad_keys
#         }
#         n_q = [
#             _get_length(quadrupolar[item])
#             for item in QUADRUPOLAR_PARAMS
#             if item in quad_keys
#         ]
#     return n_q, quad_keys, quadrupolar


# def _populate_quadrupolar(sys, items):
#     n = len(items[0])
#     for i in range(n):
#         if sys[i].sites[0].isotope.spin > 0.5:
#             sys[i].sites[0].quadrupolar = {
#                 "Cq": items[0][i],
#                 "eta": items[1][i],
#                 "alpha": items[2][i],
#                 "beta": items[3][i],
#                 "gamma": items[4][i],
#             }


# def _populate_shielding(sys, items):
#     n = len(items[0])
#     for i in range(n):
#         sys[i].sites[0].shielding_symmetric = {
#             "zeta": items[0][i],
#             "eta": items[1][i],
#             "alpha": items[2][i],
#             "beta": items[3][i],
#             "gamma": items[4][i],
#         }


# def _extend_defaults_to_list(item, n):
#     """Returns np.array if item is array-like, otherwise n-length list of item"""
#     if isinstance(item, (list, np.ndarray)):
#         return np.asarray(item)
#     return np.asarray([item for _ in range(n)])


# def _flatten_item(item):
#     """Flatten item if item is array-like, otherwise item"""
#     if isinstance(item, (list, np.ndarray)):
#         return np.asarray(item).ravel()
#     return item


# def _get_length(item):
#     """Return length of item if item is array-like, otherwise 0"""
#     if isinstance(item, (list, np.ndarray)):
#         return np.asarray(item).size
#     return 0


# def _check_size(n_list):
#     index = np.where(n_list > 0)
#     n_list_reduced = n_list[index]
#     first_item = n_list_reduced[0]
#     if np.all(n_list_reduced == first_item):
#         return first_item
#     raise ValueError(
#         "Each entry can either be a single item or a list of items. If an entry is a "
#         "list, it's length must be equal to the length of other lists present in the "
#         "system."
#     )
