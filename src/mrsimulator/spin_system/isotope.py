"""Base Isotope class."""
from os import path
from re import match
from typing import ClassVar
from typing import Dict

from monty.serialization import loadfn
from pydantic import BaseModel
from pydantic import validator

__author__ = "Deepansh Srivastava"
__email__ = "srivastava.89@osu.edu"

MODULE_DIR = path.dirname(path.abspath(__file__))
ISOTOPE_DATA = loadfn(path.join(MODULE_DIR, "isotope_data.json"))

DEFAULT_ISOTOPE = {
    "spin_multiplicity": 1,
    "gyromagnetic_ratio": 0,
    "quadrupole_moment": 0,
    "natural_abundance": 100,
    "atomic_number": 0,
}


class Isotope(BaseModel):
    """The Isotope class.

    Attributes
    ----------

    symbol: str (required)
        The isotope symbol given as the atomic number followed by the atomic symbol.

    Example
    -------

    >>> # 13C isotope information
    >>> carbon = Isotope(symbol='13C')
    >>> carbon.spin
    0.5
    >>> carbon.natural_abundance # in %
    1.11
    >>> carbon.gyromagnetic_ratio # in MHz/T
    10.708398861439887
    >>> carbon.atomic_number
    6
    >>> carbon.quadrupole_moment # in eB
    0.0
    """

    symbol: str
    spin_multiplicity: int = 1
    gyromagnetic_ratio: float = 0
    quadrupole_moment: float = 0
    natural_abundance: float = 100
    atomic_number: int = 0

    test_vars: ClassVar[Dict] = {"symbol": "1H"}
    custom_isotope_data: ClassVar[Dict] = {}

    class Config:
        extra = "forbid"
        validate_assignment = True
        allow_mutation = False

    def __init__(self, **kwargs):
        symbol = kwargs["symbol"]
        if symbol in get_all_isotope_symbols():
            kwargs_new = get_isotope_dict(symbol)
            for k, v in kwargs.items():
                if v != kwargs_new[k]:
                    raise ValueError(f"{k} for {symbol} cannot be assigned.")
        else:
            kwargs_new = DEFAULT_ISOTOPE.copy()
            kwargs_new.update(kwargs)
            Isotope.custom_isotope_data[symbol] = kwargs_new
        super().__init__(**kwargs_new)

    @validator("symbol", always=True)
    def validate_symbol(cls, v, *, values, **kwargs):
        return format_isotope_string(v)

    def json(self, **kwargs) -> dict:
        return self.dict()

    # def dict(self, **kwargs) -> dict:
    #     return self.json()

    @classmethod
    def get_isotope(cls, val):
        """Ensuring backwards compatibility with previous serializations and workflows
        means that Isotope objects may need instantiated from string values (isotope
        symbols), dictionary objects (all defining attributes of the Isotope objects),
        or Isotope objects themselves. This utility function parses the type of val and
        returns the corresponding isotope object

        Arguments:
            val: A string, dictionary, or Isotope object representing the isotope
            symbol, isotope data in dictionary form, or
            :py:class:~`mrsimulator.spin_system.isotope.Isotope` object, respectively.

        Returns:
            An instance of the Isotope class
        """
        # Return Isotope object, no further checking needed
        if isinstance(val, Isotope):
            return val

        # Check if string symbol recognized, then return Isotope object
        if isinstance(val, str):
            if val not in get_all_isotope_symbols():
                raise ValueError(f"{val} is an unrecognized Isotope symbol.")

            return Isotope(symbol=val)

        # Value is dictionary, meaning either need to add custom isotope or get symbol
        if isinstance(val, dict):
            if val["symbol"] in get_all_isotope_symbols():

                # Dictionary with only key "symbol" sometimes passed depending on
                # validation order. Here should return known isotope and skip next check
                if set(val.keys()) == {"symbol"}:
                    return Isotope(symbol=val["symbol"])

                # Ensure pre-existing data for symbol matches that of the passed dict
                if val == get_isotope_dict(val["symbol"]):
                    return Isotope(symbol=val["symbol"])

                raise ValueError(
                    "Stored isotope data does not match the provided dictionary for"
                    f" {val['symbol']}."
                )

            # Sometimes a dictionary with only key "symbol" is passed depending on
            # validation order. Here, we should check the string value
            return Isotope.add_new(**val)

        raise ValueError(f"Type {type(val)} is invalid for this method.")

    @classmethod
    def add_new(
        cls,
        symbol: str,
        spin_multiplicity: int,
        gyromagnetic_ratio: float,
        quadrupole_moment: float = 0,
        natural_abundance: float = 100,
        atomic_number: int = -1,
    ):
        """Add isotope data from a custom Isotope into the stored Isotope data and
        return an instance of the new Isotope. The isotope symbol cannot match an real
        isotope symbol; if the provided symbol matches a known custom isotope symbol,
        then an instance of that isotope is returned.

        Arguments:
            (str) symbol: Required symbol for custom isotope class. String cannot match
                another isotope symbol.
            (float) spin_multiplicity: Required spin multiplicity number for the isotope
                equivalent to (2*I + 1) where I is the spin of the isotope. Must be an
                integer.
            (float) gyromagnetic_ratio: Required gyromagnetic ratio of the isotope given
                in MHz/T.
            (float) quadrupole_moment: Optional quadrupole moment given in eB. Default
                is 0.
            (float) natural_abundance: Optional natural abundance of the isotope given
                as a percentage between 0 and 100. Default is 100.
            (int) atomic_number: Optional atomic number for the custom isotope. Can take
                any integer value and has no bering on simulated spectra.

        Returns:
            An instance of the Isotope class

        Example
        -------

        >>> # Create new custom isotope
        >>> iso = Isotope.add_new(
        ...     symbol="custom_iso",
        ...     spin_multiplicity=4,
        ...     gyromagnetic_ratio=12.345,
        ...     quadrupole_moment=-0.06789,
        ... )
        >>> iso.spin
        1.5
        >>> iso.gyromagnetic_ratio
        12.345
        >>> iso.quadrupole_moment
        -0.06789
        >>> # Create same isotope from symbol
        >>> iso2 = Isotope(symbol="custom_iso")
        >>> iso == iso2
        True
        """
        # Check for symbol overlap in dictionaries
        if symbol in ISOTOPE_DATA or symbol in Isotope.custom_isotope_data:
            raise ValueError(
                f"Symbol, {symbol}, is already attributed to another Isotope. All "
                "Isotope symbols must be unique; please choose a different symbol."
            )

        # Check for spin integer or half integer
        if not isinstance(spin_multiplicity, int) or spin_multiplicity <= 1:
            raise ValueError(
                f"Isotope spin_multiplicity value must be greater than one and must "
                f"be an integer. Got {spin_multiplicity}."
            )

        # Check abundance between 0 and 100, inclusive
        if not 0 <= natural_abundance <= 100:
            raise ValueError(
                "Abundance must be between 0 and 100, inclusive. "
                f"Got {natural_abundance}."
            )

        # Ensure atomic number is an integer value
        if not float(atomic_number).is_integer():
            raise ValueError(
                f"Atomic number must be an integer value. Got {atomic_number}"
            )

        Isotope.custom_isotope_data[symbol] = {
            "spin_multiplicity": spin_multiplicity,
            "natural_abundance": natural_abundance,
            "gyromagnetic_ratio": gyromagnetic_ratio,
            "quadrupole_moment": quadrupole_moment,
            "atomic_number": atomic_number,
        }

        return Isotope(symbol=symbol)

    @property
    def spin(self):
        """Spin quantum number, I, of the isotope."""
        isotope_data = get_isotope_dict(self.symbol)
        return (isotope_data["spin_multiplicity"] - 1) / 2.0

    # @property
    # def natural_abundance(self):
    #     """Natural abundance of the isotope in units of %."""
    #     isotope_data = get_isotope_dict(self.symbol)
    #     return isotope_data["natural_abundance"]

    # @property
    # def gyromagnetic_ratio(self):
    #     """Reduced gyromagnetic ratio of the nucleus given in units of MHz/T."""
    #     isotope_data = get_isotope_dict(self.symbol)
    #     return isotope_data["gyromagnetic_ratio"]

    # @property
    # def quadrupole_moment(self):
    #     """Quadrupole moment of the nucleus given in units of eB (electron-barn)."""
    #     isotope_data = get_isotope_dict(self.symbol)
    #     return isotope_data["quadrupole_moment"]

    # @property
    # def atomic_number(self):
    #     """Atomic number of the isotope."""
    #     isotope_data = get_isotope_dict(self.symbol)
    #     return isotope_data["atomic_number"]

    def larmor_freq(self, B0=9.4):
        """Return the Larmor frequency of the isotope at a magnetic field strength B0.

        Args:
            float B0: magnetic field strength in T

        Returns:
            float: Larmor frequency in MHz

        Example
        -------

        >>> silicon = Isotope(symbol="29Si")
        >>> freq = silicon.larmor_freq(B0 = 9.4)
        """
        return -self.gyromagnetic_ratio * B0


def format_isotope_string(isotope_string: str) -> str:
    """Format the isotope string to {A}{symbol}, where A is the isotope number."""
    # Skip formatting if the symbol is from a custom isotope
    if isotope_string in Isotope.custom_isotope_data:
        return isotope_string

    result = match(r"(\d+)\s*(\w+)", isotope_string)

    if result is None:
        raise Exception(f"Could not parse isotope string {isotope_string}")

    isotope = result.group(2)
    A = result.group(1)

    formatted_string = f"{A}{isotope}"
    if formatted_string not in ISOTOPE_DATA:
        raise Exception(f"Could not parse isotope string {isotope_string}")

    return formatted_string


def get_isotope_dict(isotope_string: str) -> dict:
    """Get the isotope's intrinsic properties from a JSON data file."""
    formatted_isotope_string = format_isotope_string(isotope_string)

    # isotope_string is always unique, and only exists in one of the dictionaries
    if formatted_isotope_string in ISOTOPE_DATA:
        isotope_dict = dict(ISOTOPE_DATA[formatted_isotope_string])
    else:
        isotope_dict = dict(Isotope.custom_isotope_data[formatted_isotope_string])

    isotope_dict.update({"symbol": formatted_isotope_string})

    return isotope_dict


def get_all_isotope_data() -> dict:
    """Return a dictionary of all isotopes, both real and custom"""
    return ISOTOPE_DATA | Isotope.custom_isotope_data


def get_all_isotope_symbols() -> list:
    """Returns a list of all currently valid isotope symbols"""
    return list(get_all_isotope_data().keys())
