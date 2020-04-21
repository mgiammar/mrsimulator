# -*- coding: utf-8 -*-
"""Base Isotope class."""
from os import path
from re import match
from typing import Optional

from monty.serialization import loadfn
from pydantic import BaseModel
from pydantic import validator

__author__ = "Deepansh J. Srivastava"
__email__ = "deepansh2012@gmail.com"

MODULE_DIR = path.dirname(path.abspath(__file__))
ISOTOPE_DATA = loadfn(path.join(MODULE_DIR, "isotope_data.json"))


class Isotope(BaseModel):
    symbol: Optional[str] = None

    class Config:
        validate_assignment = True

    @validator("symbol", always=True)
    def get_isotope(cls, v, *, values, **kwargs):
        if v is None:
            return v
        return format_isotope_string(v)

    def to_dict_with_units(self):
        return self.symbol

    @property
    def spin(self):
        """
        Spin quantum number, I, of the isotope.

        Example:
            >>> dim.spin
            2.5
        """
        if self.symbol is None:
            return None
        isotope_data = get_isotope_data(self.symbol)
        return isotope_data["spin"] / 2.0

    @property
    def natural_abundance(self):
        """
        Natural abundance of the isotope in units of %.

        Example:
            >>> dim.natural_abundance
            100.0
        """
        if self.symbol is None:
            return None
        isotope_data = get_isotope_data(self.symbol)
        return isotope_data["natural_abundance"]

    @property
    def gyromagnetic_ratio(self):
        """
        Reduced gyromagnetic ratio of the nucleus given in units of MHz/T.

        Example:
            >>> dim.gyromagnetic_ratio
            11.10309
        """
        if self.symbol is None:
            return None
        isotope_data = get_isotope_data(self.symbol)
        return isotope_data["gyromagnetic_ratio"]

    @property
    def quadrupole_moment(self):
        """
        Quadrupole moment of the nucleus given in units of eB (electron-barn).

        Example:
            >>> dim.quadrupole_moment
            0.15
        """
        if self.symbol is None:
            return None
        isotope_data = get_isotope_data(self.symbol)
        return isotope_data["quadrupole_moment"]

    @property
    def atomic_number(self):
        """
        Atomic number of the isotope.

        Example:
            >>> dim.atomic_number
            13
        """
        if self.symbol is None:
            return None
        isotope_data = get_isotope_data(self.symbol)
        return isotope_data["atomic_number"]


def format_isotope_string(isotope_string):
    """Format isotope string to {A}{symbol}, where A is the isotope number"""
    result = match(r"(\d+)\s*(\w+)", isotope_string)

    if result is None:
        raise Exception(f"Could not parse isotope string {isotope_string}")
    isotope = result.group(2)
    A = result.group(1)

    formatted_string = f"{A}{isotope}"
    if formatted_string not in ISOTOPE_DATA:
        raise Exception(f"Could not parse isotope string {isotope_string}")

    return formatted_string


def get_isotope_data(isotope_string):
    """
    Get the isotope's intrinsinc NMR properties from a JSON
    data file.
    """
    formatted_isotope_string = format_isotope_string(isotope_string)
    isotope_dict = dict(ISOTOPE_DATA[formatted_isotope_string])
    isotope_dict.update({"isotope": formatted_isotope_string})
    return isotope_dict
