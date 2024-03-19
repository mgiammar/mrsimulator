"""
Analytical czjzek ditribution on polar and non-polar grid

__author__ = "Deepansh J. Srivastava"
__email__ = "dsrivastava@hyperfine.io"
"""
import mrsimulator.models.analytical_distributions as analytical_dist
import numpy as np
from mrsimulator.spin_system.tensors import SymmetricTensor

from .utils import get_Haeberlen_components
from .utils import get_principal_components
from .utils import zeta_eta_to_x_y


__author__ = "Deepansh J. Srivastava"
__email__ = "srivastava.89@osu.edu"

ANALYTICAL_AVAILABLE = {"czjzek": analytical_dist.czjzek}


def _czjzek_random_distribution_tensors(sigma, n):
    r"""Czjzek random distribution model.

    Args:
        float sigma: The standard deviation of the five-dimensional multi-variate normal
            distribution.
        int n: Number of samples drawn from the Czjzek random distribution model.

    Description
    -----------

    U is an array of the coordinates randomly drawn from an uncorrelated
    five-dimensional multivariate normal distribution with standard deviation `sigma`
    and zero mean.

    The components of the traceless second-rank symmetric cartesian tensor, S_ij,
    follows

    Sxx = sqrt(3) * U5 - U1

    Sxy = Syx = sqrt(3) * U4

    Syy = -sqrt(3) * U5 - U1

    Sxz = Szx = sqrt(3) * U2

    Szz = 2 * U1

    Syz = Szy = sqrt(3) * U3
    """

    # The random sampling U1, U2, ... U5
    U1 = np.random.normal(0.0, sigma, n)

    sqrt_3_sigma = np.sqrt(3) * sigma
    sqrt_3_U2 = np.random.normal(0.0, sqrt_3_sigma, n)
    sqrt_3_U3 = np.random.normal(0.0, sqrt_3_sigma, n)
    sqrt_3_U4 = np.random.normal(0.0, sqrt_3_sigma, n)
    sqrt_3_U5 = np.random.normal(0.0, sqrt_3_sigma, n)

    # Create N random tensors
    tensors = np.zeros((n, 3, 3))  # n x 3 x 3 tensors

    tensors[:, 0, 0] = sqrt_3_U5 - U1  # xx
    tensors[:, 0, 1] = sqrt_3_U4  # xy
    tensors[:, 0, 2] = sqrt_3_U2  # xz

    tensors[:, 1, 0] = sqrt_3_U4  # yx
    tensors[:, 1, 1] = -sqrt_3_U5 - U1  # yy
    tensors[:, 1, 2] = sqrt_3_U3  # yz

    tensors[:, 2, 0] = sqrt_3_U2  # zx
    tensors[:, 2, 1] = sqrt_3_U3  # zy
    tensors[:, 2, 2] = 2 * U1  # zz

    return tensors


class AbstractDistribution:
    def __init__(
        self,
        mean_isotropic_chemical_shift=0.0,
        abundance=1.0,
        polar=False,
        cache_tensors=False,
    ):
        """Basic class attributes for distributions"""
        self._cache_tensors = cache_tensors
        self._tensors = None
        self.mean_isotropic_chemical_shift = mean_isotropic_chemical_shift
        self.abundance = abundance
        self.polar = polar

    def pdf(self, pos, size: int = 400000, analytical: bool = True):
        """Generates a probability distribution function by binning the random
        variates of length size onto the given grid system.

        Args:
            pos: A list of coordinates along the two dimensions given as NumPy arrays.
            size: The number of random variates drawn in generating the pdf. The default
                is 400000.

        Returns:
            A list of x and y coordinates and the corresponding amplitudes.

        Example:
            >>> import numpy as np
            >>> cq = np.arange(50) - 25
            >>> eta = np.arange(21)/20
            >>> Cq_dist, eta_dist, amp = cz_model.pdf(pos=[cq, eta])
        """
        if analytical and self.model_name in ANALYTICAL_AVAILABLE:
            analytical_model = ANALYTICAL_AVAILABLE[self.model_name]
            return analytical_model(self.sigma, pos, self.polar)
        else:
            return self.pdf_numerical(pos, size)

    def pdf_numerical(self, pos, size: int = 400000):
        """Generate distribution numerically"""
        delta_z = (pos[0][1] - pos[0][0]) / 2
        delta_e = (pos[1][1] - pos[1][0]) / 2
        x = [pos[0][0] - delta_z, pos[0][-1] + delta_z]
        y = [pos[1][0] - delta_e, pos[1][-1] + delta_e]

        x_size = pos[0].size
        y_size = pos[1].size
        zeta, eta = self.rvs(size)
        hist, _, _ = np.histogram2d(zeta, eta, bins=[x_size, y_size], range=[x, y])

        hist /= hist.sum()

        x_, y_ = np.meshgrid(pos[0], pos[1])
        return x_, y_, hist.T


class CzjzekDistribution(AbstractDistribution):
    r"""A Czjzek distribution model class.

    The Czjzek distribution model is a random sampling of second-rank traceless
    symmetric tensors whose explicit matrix form follows

    .. math::
        {\bf S} = \left[
        \begin{array}{l l l}
        \sqrt{3} U_5 - U_1   & \sqrt{3} U_4          & \sqrt{3} U_2 \\
        \sqrt{3} U_4         & -\sqrt{3} U_5 - U_1   & \sqrt{3} U_3 \\
        \sqrt{3} U_2         & \sqrt{3} U_3          & 2 U_1
        \end{array}
        \right],

    where the components, :math:`U_i`, are randomly drawn from a five-dimensional
    multivariate normal distribution. Each component, :math:`U_i`, is a dimension of
    the five-dimensional uncorrelated multivariate normal distribution with the mean
    of :math:`<U_i>=0` and the variance :math:`<U_iU_i>=\sigma^2`.

    .. math::
        S_T = S_C(\sigma),

    Args:
        float sigma: The Gaussian standard deviation.

    .. note:: In the original Czjzek paper, the parameter :math:`\sigma` is given as
        two times the standard deviation of the multi-variate normal distribution used
        here.

    Example:
        >>> from mrsimulator.models import CzjzekDistribution
        >>> cz_model = CzjzekDistribution(0.5)
    """
    model_name = "czjzek"

    def __init__(
        self,
        sigma: float,
        mean_isotropic_chemical_shift: float = 0.0,
        abundance: float = 1.0,
        polar=False,
        cache=False,
    ):
        super().__init__(
            cache_tensors=cache,
            polar=polar,
            mean_isotropic_chemical_shift=mean_isotropic_chemical_shift,
            abundance=abundance,
        )
        self.sigma = sigma

    def rvs(self, size: int):
        """Draw random variates of length `size` from the distribution.

        Args:
            size: The number of random points to draw.

        Returns:
            A list of two NumPy array, where the first and the second array are the
            anisotropic/quadrupolar coupling constant and asymmetry parameter,
            respectively.

        Example:
            >>> Cq_dist, eta_dist = cz_model.rvs(size=1000000)
        """
        if self._cache_tensors:
            if self._tensors is None:
                self._tensors = _czjzek_random_distribution_tensors(self.sigma, size)
            tensors = self._tensors
        else:
            tensors = _czjzek_random_distribution_tensors(self.sigma, size)

        if not self.polar:
            return get_Haeberlen_components(tensors)
        return zeta_eta_to_x_y(*get_Haeberlen_components(tensors))

    def param_prefix(self):
        return "czjzek"

    def get_lmfit_params(self, params, i):
        """Create lmfit params for index i"""
        params.add(f"czjzek_{i}_sigma", value=self.sigma, min=0)
        params.add(f"czjzek_{i}_iso_shift", value=self.mean_isotropic_chemical_shift)
        params.add(f"czjzek_{i}_weight", value=self.abundance, min=0, max=1)
        return params

    def update_lmfit_params(self, params, i):
        """Create lmfit params for index i"""
        prefix = self.param_prefix()

        self.sigma = params[f"{prefix}_{i}_sigma"].value
        self.eps = params[f"{prefix}_{i}_epsilon"].value
        self.mean_isotropic_chemical_shift = params[f"{prefix}_{i}_iso_shift"].value
        self.abundance = params[f"{prefix}_{i}_weight"].value


class ExtCzjzekDistribution(AbstractDistribution):
    r"""An extended Czjzek distribution distribution model.

    The extended Czjzek random distribution [#f1]_ model is an extension of the Czjzek
    model, given as

    .. math::
        S_T = S(0) + \rho S_C(\sigma=1),

    where :math:`S_T` is the total tensor, :math:`S(0)` is the dominant tensor,
    :math:`S_C(\sigma=1)` is the Czjzek random model attributing to the random
    perturbation of the tensor about the dominant tensor, :math:`S(0)`, and
    :math:`\rho` is the size of the perturbation. Note, in the above equation, the
    :math:`\sigma` parameter from the Czjzek random model, :math:`S_C`, has no meaning
    and is set to one. The factor, :math:`\rho`, is defined as

    .. math::
        \rho = \frac{||S(0)|| \epsilon}{\sqrt{30}},

    where :math:`\|S(0)\|` is the 2-norm of the dominant tensor, and :math:`\epsilon`
    is a fraction.

    .. [#f1] Gérard Le Caër, Bruno Bureau, and Dominique Massiot,
        An extension of the Czjzek model for the distributions of electric field
        gradients in disordered solids and an application to NMR spectra of 71Ga in
        chalcogenide glasses. Journal of Physics: Condensed Matter, 2010, 22, 065402.
        DOI: 10.1088/0953-8984/22/6/065402

    Args:
        SymmetricTensor symmetric_tensor: A shielding or quadrupolar symmetric tensor
            or equivalent dict object.
        float eps: A fraction determining the extent of perturbation.

    Example
    -------

    >>> from mrsimulator.models import ExtCzjzekDistribution
    >>> S0 = {"Cq": 1e6, "eta": 0.3}
    >>> ext_cz_model = ExtCzjzekDistribution(S0, eps=0.35)
    """
    model_name = "extended czjzek"

    def __init__(
        self,
        symmetric_tensor: SymmetricTensor,
        eps: float,
        mean_isotropic_chemical_shift: float = 0.0,
        abundance: float = 1.0,
        polar=False,
        cache=False,
    ):
        super().__init__(
            cache_tensors=cache,
            polar=polar,
            mean_isotropic_chemical_shift=mean_isotropic_chemical_shift,
            abundance=abundance,
        )
        if isinstance(symmetric_tensor, dict):
            self.symmetric_tensor = SymmetricTensor(**symmetric_tensor)
        else:
            self.symmetric_tensor = symmetric_tensor
        self.eps = eps

    def rvs(self, size: int):
        """Draw random variates of length `size` from the distribution.

        Args:
            size: The number of random points to draw.

        Returns:
            A list of two NumPy array, where the first and the second array are the
            anisotropic/quadrupolar coupling constant and asymmetry parameter,
            respectively.

        Example:
            >>> Cq_dist, eta_dist = ext_cz_model.rvs(size=1000000)
        """

        # czjzek_random_distribution model
        if self._cache_tensors:
            if self._tensors is None:
                self._tensors = _czjzek_random_distribution_tensors(1, size)
            tensors = self._tensors
        else:
            tensors = _czjzek_random_distribution_tensors(1, size)

        symmetric_tensor = self.symmetric_tensor

        zeta = symmetric_tensor.zeta or symmetric_tensor.Cq
        eta = symmetric_tensor.eta

        # the traceless second-rank symmetric Cartesian tensor in PAS
        T0 = [0.0, 0.0, 0.0]
        if zeta != 0 and eta != 0:
            T0 = get_principal_components(zeta, eta)

        # 2-norm of the tensor
        norm_T0 = np.linalg.norm(T0)

        # the perturbation factor
        rho = self.eps * norm_T0 / np.sqrt(30)

        # total tensor
        total_tensors = np.diag(T0) + rho * tensors

        if not self.polar:
            return get_Haeberlen_components(total_tensors)
        return zeta_eta_to_x_y(*get_Haeberlen_components(total_tensors))

    def param_prefix(self):
        return "ext_czjzek"

    def get_lmfit_params(self, params, i):
        """Create lmfit params for index i"""
        prefix = self.param_prefix()
        if self.symmetric_tensor.zeta is not None:
            zeta = self.symmetric_tensor.zeta
        else:
            zeta = self.symmetric_tensor.Cq
        params.add(f"{prefix}_{i}_zeta0", value=zeta)
        params.add(f"{prefix}_{i}_eta0", value=self.symmetric_tensor.eta, min=0, max=1)
        params.add(f"{prefix}_{i}_epsilon", value=self.eps, min=0)
        params.add(f"{prefix}_{i}_iso_shift", value=self.mean_isotropic_chemical_shift)
        params.add(f"{prefix}_{i}_weight", value=self.abundance, min=0, max=1)
        return params

    def update_lmfit_params(self, params, i):
        """Create lmfit params for index i"""
        prefix = self.param_prefix()

        zeta = params[f"{prefix}_{i}_zeta0"].value
        if self.symmetric_tensor.zeta is not None:
            self.symmetric_tensor.zeta = zeta
        else:
            self.symmetric_tensor.Cq = zeta

        self.symmetric_tensor.eta = params[f"{prefix}_{i}_eta0"].value
        self.eps = params[f"{prefix}_{i}_epsilon"].value
        self.mean_isotropic_chemical_shift = params[f"{prefix}_{i}_iso_shift"].value
        self.abundance = params[f"{prefix}_{i}_weight"].value
