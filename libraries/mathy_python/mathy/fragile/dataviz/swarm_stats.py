"""Contains streaming plots that visualize the internal data of a :class:`Swarm`."""
from typing import Tuple
import warnings

import holoviews
from holoviews import Store
import numpy
import pandas

from fragile.core.bounds import Bounds
from fragile.core.functions import relativize
from fragile.core.swarm import Swarm
from fragile.core.utils import get_plangym_env
from fragile.dataviz.streaming import Curve, Histogram, Landscape2D, RGB, Table

PLOT_NAMES = ()


class SummaryTable(Table):
    """
    Display a table containing information about the current algorithm run.

    The displayed information contains:
        - Epoch: Number of the current epoch of the :class:`Swarm`.
        - Best Reward: Value of the best reward found in the current run.
        - Deaths: Percentage of walkers that fell out of the domain's boundary \
                  condition during the last iteration.
        - Clones: Percentage of walkers that cloned during the last iteration.
    """

    name = "summary_table"

    def __init__(self, mpl_opts: dict = None, bokeh_opts: dict = None, *args, **kwargs):
        """
        Initialize a :class:`SummaryTable`.

        Args:
            bokeh_opts: Default options for the plot when rendered using the \
                       "bokeh" backend.
            mpl_opts: Default options for the plot when rendered using the \
                    "matplotlib" backend.
            *args: Passed to :class:`Table`.__init__
            **kwargs: Passed to :class:`Table`.__init__
        """
        default_bokeh_opts = {"width": 350}
        default_mpl_opts = {}
        mpl_opts, bokeh_opts = self.update_default_opts(
            default_mpl_opts, mpl_opts, default_bokeh_opts, bokeh_opts
        )
        super(SummaryTable, self).__init__(
            mpl_opts=mpl_opts, bokeh_opts=bokeh_opts, *args, **kwargs
        )

    def opts(self, title="Run summary", *args, **kwargs):
        """
        Update the plot parameters. Same as ``holoviews`` ``opts``.

        The default values updates the plot axes independently when being \
        displayed in a :class:`Holomap`.
        """
        plot_opts = self.update_kwargs(**kwargs)
        super(SummaryTable, self).opts(title=title, *args, **plot_opts)

    def get_plot_data(self, swarm: Swarm = None):
        """Extract the best reward found by the swarm and create a \
        :class:`pandas.DataFrame` to keep track of it."""
        columns = ["Epoch", "Best Reward", "Deaths", "Clones"]
        if swarm is None:
            data = pandas.DataFrame(
                {"Epoch": [], "Best Reward": [], "Deaths": [], "Clones": []}, columns=columns
            )
        else:
            oobs = swarm.get("oobs")
            will_clone = swarm.get("will_clone")
            best_reward = swarm.get("best_reward")
            epoch = swarm.get("epoch")
            deaths = float(oobs.sum()) / len(swarm)
            clones = float(will_clone.sum()) / len(swarm)
            data = pandas.DataFrame(
                {
                    "Epoch": [int(epoch)],
                    "Best Reward": ["{:.4f}".format(float(best_reward))],
                    "Deaths": ["{:.2f}%".format(100 * deaths)],
                    "Clones": ["{:.2f}%".format(100 * clones)],
                },
                columns=columns,
            )
        return data


class AtariBestFrame(RGB):
    """
    Display the Atari frame that corresponds to the best state sampled.

    It only works for environments of class :class:`AtariEnv`.
    """

    name = "best_frame"

    def __init__(self, mpl_opts: dict = None, bokeh_opts: dict = None, *args, **kwargs):
        """
        Initialize a :class:`AtariBestFrame`.

        Args:
            bokeh_opts: Default options for the plot when rendered using the \
                       "bokeh" backend.
            mpl_opts: Default options for the plot when rendered using the \
                    "matplotlib" backend.
            *args: Passed to :class:`RGB`.__init__
            **kwargs: Passed to :class:`RGB`.__init__
        """
        default_bokeh_opts = {"width": 160, "height": 210}
        default_mpl_opts = {}
        mpl_opts, bokeh_opts = self.update_default_opts(
            default_mpl_opts, mpl_opts, default_bokeh_opts, bokeh_opts
        )
        super(AtariBestFrame, self).__init__(
            mpl_opts=mpl_opts, bokeh_opts=bokeh_opts, *args, **kwargs
        )

    @staticmethod
    def image_from_state(swarm: Swarm, state: numpy.ndarray) -> numpy.ndarray:
        """
        Return the frame corresponding to a given :class:`AtariEnv` state.

        Args:
            swarm: Swarm containing the target environment.
            state: States that will be used to extract the frame.

        Returns:
            Array of size (210, 160, 3) containing the RGB frame representing \
            the target state.

        """
        env = get_plangym_env(swarm)
        env.set_state(state.astype(numpy.uint8).copy())
        env.step(0)
        return numpy.asarray(env.ale.getScreenRGB(), dtype=numpy.uint8)

    def get_plot_data(self, swarm: Swarm = None) -> numpy.ndarray:
        """Extract the frame from the :class:`AtariEnv` that the target \
        :class:`Swarm` contains."""
        if swarm is None:
            return numpy.zeros((210, 160, 3))
        state = swarm.get("best_state")
        return self.image_from_state(swarm=swarm, state=state)

    def opts(self, title="Best state sampled", *args, **kwargs):
        """
        Update the plot parameters. Same as ``holoviews`` ``opts``.

        The default values updates the plot axes independently when being \
        displayed in a :class:`Holomap`.
        """
        kwargs = self.update_kwargs(**kwargs)
        super(AtariBestFrame, self).opts(title=title, *args, **kwargs)


class BestReward(Curve):
    """Plot a curve that represents the evolution of the best reward found."""

    name = "best_reward"

    def opts(
        self,
        title="Best value found",
        xlabel: str = "iteration",
        ylabel: str = "Best value",
        *args,
        **kwargs
    ):
        """
        Update the plot parameters. Same as ``holoviews`` ``opts``.

        The default values updates the plot axes independently when being \
        displayed in a :class:`Holomap`.
        """
        super(BestReward, self).opts(title=title, xlabel=xlabel, ylabel=ylabel, *args, **kwargs)

    def get_plot_data(self, swarm: Swarm = None):
        """Extract the best reward found by the swarm and create a \
        :class:`pandas.DataFrame` to keep track of it."""
        if swarm is None:
            data = pandas.DataFrame({"x": [], "best_val": []}, columns=["x", "best_val"])
        else:

            data = pandas.DataFrame(
                {"x": [int(swarm.get("epoch"))], "best_val": [float(swarm.get("best_reward"))]},
                columns=["x", "best_val"],
            )
        return data


class SwarmHistogram(Histogram):
    """Abstract class to create histograms using data from the :class:`Swarm`."""

    name = "swarm_histogram"

    def __init__(self, margin_high=1.0, margin_low=1.0, *args, **kwargs):
        """
        Initialize a :class:`SwarmHistogram`.

        Args:
            margin_high: Ignored. (Update pending)
            margin_low: Ignored. (Update pending)
            *args: Passed to :class:`Histogram`.
            **kwargs: Passed to :class:`Histogram`.
        """
        self.high = margin_high
        self.low = margin_low
        super(SwarmHistogram, self).__init__(*args, **kwargs)

    def opts(self, ylabel: str = "Frequency", *args, **kwargs):
        """
        Update the plot parameters. Same as ``holoviews`` ``opts``.

        The default values updates the plot axes independently when being \
        displayed in a :class:`Holomap`.
        """
        super(SwarmHistogram, self).opts(ylabel=ylabel, *args, **kwargs)

    def _update_lims(self, X: numpy.ndarray):
        """Update the x axis boundaries to match the values of the plotted distribution."""
        self.xlim = (X.min(), X.max())

    def get_plot_data(self, swarm: Swarm, attr: str):
        """
        Extract the data of the attribute of the :class:`Swarm` that will be \
        represented as a histogram.

        Args:
            swarm: Target :class:`Swarm`.
            attr: Attribute of the target :class:`States` that will be plotted.

        Returns:
            Histogram containing the target data.

        """
        if swarm is None:
            return super(SwarmHistogram, self).get_plot_data(swarm)
        data = swarm.get(attr) if swarm is not None else numpy.arange(10)
        self._update_lims(data)
        return super(SwarmHistogram, self).get_plot_data(data)


class RewardHistogram(SwarmHistogram):
    """Plot a histogram of the reward distribution of a :class:`Swarm`."""

    name = "reward_histogram"

    def opts(self, title="Reward distribution", xlabel: str = "Reward", *args, **kwargs):
        """
        Update the plot parameters. Same as ``holoviews`` ``opts``.

        The default values updates the plot axes independently when being \
        displayed in a :class:`Holomap`.
        """
        super(RewardHistogram, self).opts(title=title, xlabel=xlabel, *args, **kwargs)

    def get_plot_data(self, swarm: Swarm):
        """Return a histogram of the ``cum_rewards`` attribute of the :class:`Walkers`."""
        return super(RewardHistogram, self).get_plot_data(swarm, "cum_rewards")


class DistanceHistogram(SwarmHistogram):
    """Plot a histogram of the distance distribution of a :class:`Swarm`."""

    name = "distance_histogram"

    def opts(
        self,
        title="Distance distribution",
        xlabel: str = "Distance",
        ylabel: str = "Frequency",
        *args,
        **kwargs
    ):
        """
        Update the plot parameters. Same as ``holoviews`` ``opts``.

        The default values updates the plot axes independently when being \
        displayed in a :class:`Holomap`.
        """
        super(DistanceHistogram, self).opts(
            title=title, xlabel=xlabel, ylabel=ylabel, *args, **kwargs
        )

    def get_plot_data(self, swarm: Swarm):
        """Return a histogram of the ``distance`` attribute of the :class:`Walkers`."""
        return super(DistanceHistogram, self).get_plot_data(swarm, "distances")


class VirtualRewardHistogram(SwarmHistogram):
    """Plot a histogram of the virtual reward distribution of a :class:`Swarm`."""

    name = "virtual_reward_histogram"

    def opts(
        self,
        title="Virtual reward distribution",
        xlabel: str = "Virtual reward",
        ylabel: str = "Frequency",
        *args,
        **kwargs
    ):
        """
        Update the plot parameters. Same as ``holoviews`` ``opts``.

        The default values updates the plot axes independently when being \
        displayed in a :class:`Holomap`.
        """
        super(VirtualRewardHistogram, self).opts(
            title=title, xlabel=xlabel, ylabel=ylabel, *args, **kwargs
        )

    def get_plot_data(self, swarm: Swarm):
        """Return a histogram of the ``virtual_reward`` attribute of the :class:`Walkers`."""
        return super(VirtualRewardHistogram, self).get_plot_data(swarm, "virtual_rewards")


def has_embedding(swarm: Swarm) -> bool:
    """Return ``True`` if the target :class:`Swarm` has a critic to calculate embeddings."""
    if hasattr(swarm, "critic"):
        if hasattr(swarm.critic, "preprocess_input"):
            return True
    return False


def get_xy_coords(swarm: Swarm, use_embedding: False) -> numpy.ndarray:
    """
    Get the x,y coordinates to represent observations values in 2D.

    If the :class:`Swarm` has a critic to calculate 2D embeddings, the \
    two first dimensions of embedding values will be used. Otherwise return \
    the first two dimensions of ``observs``.
    """
    if use_embedding and has_embedding(swarm):
        X = swarm.critic.preprocess_input(
            env_states=swarm.walkers.env_states,
            walkers_states=swarm.walkers.states,
            model_states=swarm.walkers.model_states,
            batch_size=swarm.walkers.n,
        )
        return X[:, :2]
    elif isinstance(swarm, numpy.ndarray):
        return swarm
    return swarm.walkers.env_states.observs[:, :2]


class SwarmLandscape(Landscape2D):
    """Plot a 2D landscape of :class:`Swarm` data."""

    name = "swarm_landscape"

    def __init__(
        self, use_embeddings: bool = True, margin_high=1.0, margin_low=1.0, *args, **kwargs
    ):
        """
        Initialize a :class:`SwarmLandscape2D`.

        Args:
            use_embeddings: Use embeddings to represent the observations if available.
            margin_high: Ignored. (Update pending)
            margin_low: Ignored. (Update pending)
            *args: Passed to :class:`Landscape2D`.
            **kwargs: Passed to :class:`Landscape2D`.
        """
        self.use_embeddings = use_embeddings
        self.high = margin_high
        self.low = margin_low
        super(SwarmLandscape, self).__init__(*args, **kwargs)

    def get_z_coords(self, swarm: Swarm, X: numpy.ndarray) -> numpy.ndarray:
        """Calculate the z values of the target lanscape."""
        raise NotImplementedError

    def _update_lims(self, swarm, X: numpy.ndarray):
        backup_bounds = swarm.env.bounds if swarm is not None else Bounds.from_array(X)
        bounds = (
            swarm.critic.bounds if has_embedding(swarm) and self.use_embeddings else backup_bounds
        )
        self.xlim, self.ylim = bounds.to_tuples()[:2]

    def opts(self, xlim="default", ylim="default", *args, **kwargs):
        """
        Update the plot parameters. Same as ``holoviews`` ``opts``.

        The default values updates the plot axes independently when being \
        displayed in a :class:`Holomap`.
        """
        xlim = self.xlim if xlim == "default" else xlim
        ylim = self.ylim if ylim == "default" else ylim
        return super(SwarmLandscape, self).opts(xlim=xlim, ylim=ylim, *args, **kwargs)

    def _get_plot_data_with_defaults(self, swarm: Swarm) -> Tuple:
        if swarm is not None:
            X = get_xy_coords(swarm, self.use_embeddings)
            z = self.get_z_coords(swarm, X)
            self.invert_cmap = swarm.walkers.minimize
            self._update_lims(swarm, X)
        else:
            X = numpy.random.standard_normal((10, 2))
            z = numpy.random.standard_normal(10)
        return X, z

    def get_plot_data(self, swarm: Swarm) -> numpy.ndarray:
        """Extract the observations and target values needed to plot the landscape."""
        X, z = self._get_plot_data_with_defaults(swarm)
        data = X[:, 0], X[:, 1], z
        return super(SwarmLandscape, self).get_plot_data(data)


class RewardLandscape(SwarmLandscape):
    """Plot an interpolate landscape of the reward distribution of the :class:`Walkers`."""

    name = "reward_landscape"

    def opts(self, title="Reward landscape", *args, **kwargs):
        """
        Update the plot parameters. Same as ``holoviews`` ``opts``.

        The default values updates the plot axes independently when being \
        displayed in a :class:`Holomap`.
        """
        return super(RewardLandscape, self).opts(title=title, *args, **kwargs)

    def get_z_coords(self, swarm: Swarm, X: numpy.ndarray = None):
        """Return the normalized ``cum_rewards`` of the walkers."""
        rewards: numpy.ndarray = relativize(swarm.get("cum_rewards"))
        return rewards


class VirtualRewardLandscape(SwarmLandscape):
    """Plot an interpolate landscape of the virtual reward distribution of the :class:`Walkers`."""

    name = "virtual_reward_landscape"

    def opts(self, title="Virtual reward landscape", *args, **kwargs):
        """
        Update the plot parameters. Same as ``holoviews`` ``opts``.

        The default values updates the plot axes independently when being \
        displayed in a :class:`Holomap`.
        """
        super(VirtualRewardLandscape, self).opts(title=title, *args, **kwargs)

    def get_z_coords(self, swarm: Swarm, X: numpy.ndarray = None):
        """Return the normalized ``virtual_rewards`` of the walkers."""
        virtual_rewards: numpy.ndarray = swarm.get("virtual_rewards")
        return virtual_rewards


class DistanceLandscape(SwarmLandscape):
    """Plot an interpolate landscape of the distance distribution of the :class:`Walkers`."""

    name = "distance_landscape"

    def opts(self, title="Distance landscape", *args, **kwargs):
        """
        Update the plot parameters. Same as ``holoviews`` ``opts``.

        The default values updates the plot axes independently when being \
        displayed in a :class:`Holomap`.
        """
        super(DistanceLandscape, self).opts(title=title, *args, **kwargs)

    def get_z_coords(self, swarm: Swarm, X: numpy.ndarray = None):
        """Return the normalized ``distances`` of the walkers."""
        distances: numpy.ndarray = swarm.get("distances")
        return distances


class WalkersDensity(SwarmLandscape):
    """Plot the density distribution of the walkers."""

    name = "walkers_density"

    def __init__(
        self,
        use_embeddings: bool = True,
        mpl_opts: dict = None,
        bokeh_opts: dict = None,
        *args,
        **kwargs
    ):
        """
        Initialize a :class:`WalkersDensity`.

        Args:
            use_embeddings: Use embeddings to represent the observations if available.
            bokeh_opts: Default options for the plot when rendered using the \
                       "bokeh" backend.
            mpl_opts: Default options for the plot when rendered using the \
                    "matplotlib" backend.
            *args: Passed to :class:`SwarmLandscape`.
            **kwargs: Passed to :class:`SwarmLandscape`.
        """
        self.use_embeddings = use_embeddings
        default_bokeh_opts = {
            "height": 350,
            "width": 400,
            "shared_axes": False,
            "tools": ["hover"],
        }
        default_mpl_opts = {}
        mpl_opts, bokeh_opts = self.update_default_opts(
            default_mpl_opts, mpl_opts, default_bokeh_opts, bokeh_opts
        )
        super(WalkersDensity, self).__init__(
            mpl_opts=mpl_opts, bokeh_opts=bokeh_opts, *args, **kwargs
        )

    def get_z_coords(self, swarm: Swarm, X: numpy.ndarray):
        """Do nothing."""
        pass

    def opts(
        self,
        title="Walkers density distribution",
        xlabel: str = "x",
        ylabel: str = "y",
        framewise: bool = True,
        axiswise: bool = True,
        normalize: bool = True,
        *args,
        **kwargs
    ):
        """
        Update the plot parameters. Same as ``holoviews`` ``opts``.

        The default values updates the plot axes independently when being \
        displayed in a :class:`Holomap`.
        """
        kwargs = self.update_kwargs(**kwargs)
        # Add specific defaults to Scatter
        scatter_kwargs = dict(kwargs)
        if Store.current_backend == "bokeh":
            scatter_kwargs["fill_color"] = scatter_kwargs.get("fill_color", "red")
            scatter_kwargs["size"] = scatter_kwargs.get("size", 3.5)
        elif Store.current_backend == "matplotlib":
            scatter_kwargs["color"] = scatter_kwargs.get("color", "red")
            scatter_kwargs["s"] = scatter_kwargs.get("s", 15)
        self.plot = self.plot.opts(
            holoviews.opts.Bivariate(
                title=title,
                xlabel=xlabel,
                ylabel=ylabel,
                framewise=framewise,
                axiswise=axiswise,
                normalize=normalize,
                show_legend=False,
                colorbar=True,
                filled=True,
                *args,
                **kwargs
            ),
            holoviews.opts.Scatter(
                alpha=0.7,
                xlabel=xlabel,
                ylabel=ylabel,
                framewise=framewise,
                axiswise=axiswise,
                normalize=normalize,
                *args,
                **scatter_kwargs
            ),
            holoviews.opts.NdOverlay(normalize=normalize, framewise=framewise, axiswise=axiswise,),
        )

    def get_plot_data(self, swarm: Swarm) -> Tuple:
        """Return the observations of the walkers and the plot limits."""
        X, z = self._get_plot_data_with_defaults(swarm)
        return X, X[:, 0], X[:, 1], self.xlim, self.ylim

    @staticmethod
    def plot_landscape(data):
        """Plot the walkers distribution overlaying a histogram on a bivariate plot."""
        X, x, y, xlim, ylim = data
        mesh = holoviews.Bivariate(X)
        scatter = holoviews.Scatter((x, y))
        contour_mesh = mesh * scatter
        return contour_mesh.redim(
            x=holoviews.Dimension("x", range=xlim), y=holoviews.Dimension("y", range=ylim),
        )


class GridLandscape(SwarmLandscape):
    """Plot the memory grid of a :class:`Critic` that provides a memory grid."""

    name = "grid_landscape"

    def opts(self, title="Memory grid values", *args, **kwargs):
        """
        Update the plot parameters. Same as ``holoviews`` ``opts``.

        The default values updates the plot axes independently when being \
        displayed in a :class:`Holomap`.
        """
        super(GridLandscape, self).opts(title=title, *args, **kwargs)

    @staticmethod
    def plot_landscape(data):
        """Overlay the walkers on top of the bins defined by the memory grid."""
        x, y, xx, yy, z, xlim, ylim = data
        try:
            memory_vals = z.reshape(xx.shape)
        except ValueError:  # Avoid errors when initializing the plot.
            memory_vals = numpy.ones_like(xx)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mesh = holoviews.QuadMesh((xx, yy, memory_vals))
            plot = (mesh * holoviews.Scatter((x, y))).opts(xlim=xlim, ylim=ylim)
        return plot

    def get_z_coords(self, swarm: Swarm, X: numpy.ndarray = None) -> numpy.ndarray:
        """Extract the memory values of the :class:`Critic`'s grid."""
        if swarm is None:
            return numpy.ones(self.n_points ** self.n_points)
        if swarm.critic.bounds is None:
            swarm.critic.bounds = Bounds.from_array(X, scale=1.1)
        # target grid to interpolate to
        xi = numpy.linspace(swarm.critic.bounds.low[0], swarm.critic.bounds.high[0], self.n_points)
        yi = numpy.linspace(swarm.critic.bounds.low[1], swarm.critic.bounds.high[1], self.n_points)
        xx, yy = numpy.meshgrid(xi, yi)
        grid = numpy.c_[xx.ravel(), yy.ravel()]
        if swarm.swarm.critic.warmed:
            memory_values = swarm.swarm.critic.model.transform(grid)
            memory_values = numpy.array(
                [
                    swarm.swarm.critic.memory[ix[0], ix[1]].astype(numpy.float32)
                    for ix in memory_values.astype(int)
                ]
            )
        else:
            memory_values = numpy.arange(grid.shape[0])
        return memory_values


class KDELandscape(SwarmLandscape):
    """Plot the probability landscape of a :class:`Critic` that provides a KDE memory."""

    name = "kde_landscape"

    def opts(self, title="Memory grid values", *args, **kwargs):
        """
        Update the plot parameters. Same as ``holoviews`` ``opts``.

        The default values updates the plot axes independently when being \
        displayed in a :class:`Holomap`.
        """
        super(KDELandscape, self).opts(title=title, *args, **kwargs)

    @staticmethod
    def plot_landscape(data):
        """Display the walkers overlied on a mesh representing the probability \
        assigned by the :class:`Critic`."""
        x, y, xx, yy, z, xlim, ylim = data
        try:
            memory_vals = z.reshape(xx.shape)
        except ValueError:  # Avoid errors when initializing the plot.
            memory_vals = numpy.ones_like(xx)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mesh = holoviews.QuadMesh((xx, yy, memory_vals))
        plot = (mesh * holoviews.Scatter((x, y))).opts(xlim=xlim, ylim=ylim)
        return plot

    def get_z_coords(self, swarm: Swarm, X: numpy.ndarray = None):
        """Get the values assigned by the :class:`Critic` to the regions of the state space."""
        if swarm is None:
            return numpy.ones(self.n_points ** self.n_points)
        if swarm.critic.bounds is None:
            swarm.critic.bounds = Bounds.from_array(X, scale=1.1)
        # target grid to interpolate to
        xi = numpy.linspace(swarm.critic.bounds.low[0], swarm.critic.bounds.high[0], self.n_points)
        yi = numpy.linspace(swarm.critic.bounds.low[1], swarm.critic.bounds.high[1], self.n_points)
        xx, yy = numpy.meshgrid(xi, yi)
        grid = numpy.c_[xx.ravel(), yy.ravel()]
        if swarm.swarm.critic.warmed:
            memory_values = swarm.swarm.critic.predict(grid)
            memory_values = relativize(-memory_values)
        else:
            memory_values = numpy.arange(grid.shape[0])
        return memory_values
