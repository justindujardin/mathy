from collections.abc import Iterable as _Iterable
from typing import Iterable, Optional, Tuple, Union

import numpy as numpy

from fragile.core.utils import Scalar


class Bounds:
    """
    The :class:`Bounds` implements the logic for defining and managing closed intervals, \
    and checking if a numpy array's values are inside a given interval.

    It is used on a numpy array of a target shape.
    """

    def __init__(
        self,
        high: Union[numpy.ndarray, Scalar] = numpy.inf,
        low: Union[numpy.ndarray, Scalar] = numpy.NINF,
        shape: Optional[tuple] = None,
        dtype: Optional[type] = None,
    ):
        """
        Initialize a :class:`Bounds`.

        Args:
            high: Higher value for the bound interval. If it is an scalar it will be applied to \
                  all the coordinates of a target vector. If it is a vector, the bounds will be \
                  checked coordinate-wise. It defines and closed interval.
            low: Lower value for the bound interval. If it is an scalar it will be applied to \
                  all the coordinates of a target vector. If it is a vector, the bounds will be \
                  checked coordinate-wise. It defines and closed interval.
            shape: Shape of the array that will be bounded. Only needed if `high` and `low` are \
                   vectors and it is used to define the dimensions that will be bounded.
            dtype:  Data type of the array that will be bounded. It can be inferred from `high` \
                    or `low` (the type of `high` takes priority).

        Examples:
            Initializing :class:`Bounds` using  numpy arrays:

            >>> import numpy
            >>> high, low = numpy.ones(3, dtype=float), -1 * numpy.ones(3, dtype=int)
            >>> bounds = Bounds(high=high, low=low)
            >>> print(bounds)
            Bounds shape float64 dtype (3,) low [-1 -1 -1] high [1. 1. 1.]

            Initializing :class:`Bounds` using  scalars:

            >>> import numpy
            >>> high, low, shape = 4, 2.1, (5,)
            >>> bounds = Bounds(high=high, low=low, shape=shape)
            >>> print(bounds)
            Bounds shape float64 dtype (5,) low [2.1 2.1 2.1 2.1 2.1] high [4. 4. 4. 4. 4.]

        """
        # Infer shape if not specified
        if shape is None and hasattr(high, "shape"):
            shape = high.shape
        elif shape is None and hasattr(low, "shape"):
            shape = low.shape
        elif shape is None:
            raise TypeError("If shape is None high or low need to have .shape attribute.")
        # High and low will be arrays of target shape
        if not isinstance(high, numpy.ndarray):
            high = numpy.array(high) if isinstance(high, _Iterable) else numpy.ones(shape) * high
        if not isinstance(low, numpy.ndarray):
            low = numpy.array(low) if isinstance(low, _Iterable) else numpy.ones(shape) * low
        self.high = high
        self.low = low
        if dtype is not None:
            self.dtype = dtype
        elif hasattr(high, "dtype"):
            self.dtype = high.dtype
        elif hasattr(low, "dtype"):
            self.dtype = low.dtype
        else:
            self.dtype = type(high) if high is not None else type(low)

    def __repr__(self):
        return "{} shape {} dtype {} low {} high {}".format(
            self.__class__.__name__, self.dtype, self.shape, self.low, self.high
        )

    def __len__(self) -> int:
        """Return the number of dimensions of the bounds."""
        return len(self.high)

    @property
    def shape(self) -> Tuple:
        """
        Get the shape of the current bounds.

        Returns:
            tuple containing the shape of `high` and `low`

        """
        return self.high.shape

    @classmethod
    def from_tuples(cls, bounds: Iterable[tuple]) -> "Bounds":
        """
        Instantiate a :class:`Bounds` from a collection of tuples containing \
        the higher and lower bounds for every dimension as a tuple.

        Args:
            bounds: Iterable that returns tuples containing the higher and lower \
                    bound for every dimension of the target bounds.

        Returns:
                :class:`Bounds` instance.

        Examples:
            >>> intervals = ((-1, 1), (-2, 1), (2, 3))
            >>> bounds = Bounds.from_tuples(intervals)
            >>> print(bounds)
            Bounds shape int64 dtype (3,) low [-1 -2  2] high [1 1 3]

        """
        low, high = [], []
        for lo, hi in bounds:
            low.append(lo)
            high.append(hi)
        low, high = numpy.array(low), numpy.array(high)
        return Bounds(low=low, high=high)

    @staticmethod
    def get_scaled_intervals(
        low: Union[numpy.ndarray, float, int], high: Union[numpy.ndarray, float, int], scale: float
    ) -> Tuple[Union[numpy.ndarray, float], Union[numpy.ndarray, float]]:
        """
        Scale the high and low vectors by an scale factor.

        The value of the high and low will be proportional to the maximum and minimum values of \
        the array. Scale defines the proportion to make the bounds bigger and smaller. For \
        example, if scale is 1.1 the higher bound will be 10% higher, and the lower bounds 10% \
        smaller. If scale is 0.9 the higher bound will be 10% lower, and the lower bound 10% \
        higher. If scale is one, `high` and `low` will be equal to the maximum and minimum values \
        of the array.

        Args:
            high: Higher bound to be scaled.
            low: Lower bound to be scaled.
            scale: Value representing the tolerance in percentage from the current maximum and \
            minimum values of the array.

        Returns:
            :class:`Bounds` instance.

        """
        pct = scale - 1
        big_scale = 1 + numpy.abs(pct)
        small_scale = 1 - numpy.abs(pct)
        if pct > 0:
            xmin_scaled = numpy.where(low < 0, low * big_scale, low * small_scale)
            xmax_scaled = numpy.where(high < 0, high * small_scale, high * big_scale)
        else:
            xmin_scaled = numpy.where(low < 0, low * small_scale, low * small_scale)
            xmax_scaled = numpy.where(high < 0, high * big_scale, high * small_scale)
        return xmin_scaled, xmax_scaled

    @classmethod
    def from_array(cls, x: numpy.ndarray, scale: float = 1.0) -> "Bounds":
        """
        Instantiate a bounds compatible for bounding the given array. It also allows to set a \
        margin for the high and low values.

        The value of the high and low will be proportional to the maximum and minimum values of \
        the array. Scale defines the proportion to make the bounds bigger and smaller. For \
        example, if scale is 1.1 the higher bound will be 10% higher, and the lower bounds 10% \
        smaller. If scale is 0.9 the higher bound will be 10% lower, and the lower bound 10% \
        higher. If scale is one, `high` and `low` will be equal to the maximum and minimum values \
        of the array.

        Args:
            x: Numpy array used to initialize the bounds.
            scale: Value representing the tolerance in percentage from the current maximum and \
            minimum values of the array.

        Returns:
            :class:`Bounds` instance.

        Examples:
            >>> import numpy
            >>> x = numpy.ones((3, 3))
            >>> x[1:-1, 1:-1] = -5
            >>> bounds = Bounds.from_array(x, scale=1.5)
            >>> print(bounds)
            Bounds shape float64 dtype (3,) low [ 0.5 -7.5  0.5] high [1.5 1.5 1.5]

        """
        xmin, xmax = x.min(axis=0), x.max(axis=0)
        xmin_scaled, xmax_scaled = cls.get_scaled_intervals(xmin, xmax, scale)
        return Bounds(low=xmin_scaled, high=xmax_scaled)

    def clip(self, x: numpy.ndarray) -> numpy.ndarray:
        """
        Clip the values of the target array to fall inside the bounds (closed interval).

        Args:
            x: Numpy array to be clipped.

        Returns:
            Clipped numpy array with all its values inside the defined bounds.

        """
        return numpy.clip(x, self.low, self.high)

    def points_in_bounds(self, x: numpy.ndarray) -> Union[numpy.ndarray, bool]:
        """
        Check if the rows of the target array have all their coordinates inside \
        specified bounds.

        If the array is one dimensional it will return a boolean, otherwise a vector of booleans.

        Args:
            x: Array to be checked against the bounds.

        Returns:
            Numpy array of booleans indicating if a row lies inside the bounds.

        """
        match = self.clip(x) == x
        return match.all(axis=1).flatten() if len(match.shape) > 1 else match.all()

    def safe_margin(
        self,
        low: Union[numpy.ndarray, float] = None,
        high: Optional[Union[numpy.ndarray, float]] = None,
        scale: float = 1.0,
    ) -> "Bounds":
        """
        Initialize a new :class:`Bounds` with its bounds increased o decreased \
        by an scale factor.

        This is done multiplying both high and low for a given factor. The value of the new \
        high and low will be proportional to the maximum and minimum values of \
        the array. Scale defines the proportion to make the bounds bigger and smaller. For \
        example, if scale is 1.1 the higher bound will be 10% higher, and the lower bounds 10% \
        smaller. If scale is 0.9 the higher bound will be 10% lower, and the lower bound 10% \
        higher. If scale is one, `high` and `low` will be equal to the maximum and minimum values \
        of the array.

        Args:
            high: Used to scale the `high` value of the current instance.
            low: Used to scale the `low` value of the current instance.
            scale: Value representing the tolerance in percentage from the current maximum and \
            minimum values of the array.

        Returns:
            :class:`Bounds` with scaled high and low values.

        """
        xmax = self.high if high is None else high
        xmin = self.low if low is None else low
        xmin_scaled, xmax_scaled = self.get_scaled_intervals(xmin, xmax, scale)
        return Bounds(low=xmin_scaled, high=xmax_scaled)

    def to_tuples(self) -> Tuple[Tuple[Scalar, Scalar], ...]:
        """
        Return a tuple of tuples containing the lower and higher bound for each \
        coordinate of the :class:`Bounds` shape.

        Returns:
            Tuple of the form ((x0_low, x0_high), (x1_low, x1_high), ...,\
              (xn_low, xn_high))
        Examples:
            >>> import numpy
            >>> array = numpy.array([1, 2, 5])
            >>> bounds = Bounds(high=array, low=-array)
            >>> print(bounds.to_tuples())
            ((-1, 1), (-2, 2), (-5, 5))

        """
        return tuple([dim for dim in zip(self.low, self.high)])
