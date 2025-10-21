import numpy as np
import matplotlib.patches as patches
import matplotlib.pyplot as plt

from IPython.display import display
import ipywidgets as widgets
import IPython


class WavecalWidget(widgets.HBox):
    """
    Interactive calibration that can help users perform a
    wavelength calibration.

    Parameters:
    -----------
    upper_lines :: list
        The spatial locations of each line as found in your
        arc lamp calibration exposure.
    lower_lines :: list
        The known wavelengths of emission lines for your given
        arc lamp. This list does not need to be the same length
        as your spatial lines.
    upper_intensities :: list
        (OPTIONAL) The relative intensities of the lines found
        in your arc lamp calibration exposure. If no list is
        provided, all lines will be shown with equal intensity.
    lower_intensities :: list
        (OPTIONAL) The relative intensities of the known emissions
        for your arc lamp. If no list is provided, all lines will
        be shown with equal intensity.
    """

    def __init__(
        self, upper_lines, lower_lines, upper_intensities=None, lower_intensities=None
    ):

        super().__init__()

        # Helpful functino for normalizing relative intensities
        def norm_intensities(intensities, lines):
            if intensities is not None:
                return intensities / np.max(intensities)
            else:
                return np.ones(len(lines))

        # Normalizes intensities
        upper_intensities = norm_intensities(upper_intensities, upper_lines)
        lower_intensities = norm_intensities(lower_intensities, lower_lines)

        # Sorts all line lists by increasing position
        u_sort_idx = np.argsort(upper_lines)
        l_sort_idx = np.argsort(lower_lines)
        u_lines_sorted = np.array(upper_lines)[u_sort_idx]
        l_lines_sorted = np.array(lower_lines)[l_sort_idx]
        u_int_sorted = np.array(upper_intensities)[u_sort_idx]
        l_int_sorted = np.array(lower_intensities)[l_sort_idx]

        # Stores a copy of original arrays
        self.original_lines = [u_lines_sorted, l_lines_sorted]
        self.original_intensities = [u_int_sorted, l_int_sorted]

        # Default class values
        self.fit_order = 3
        self.res_size = 5
        self.gauss_stds = [1, 1]
        self.images = [None, None]
        self.active = False
        self.final_lines = None
        self.interaction = "Remove"
        self.mode_idx = 0
        self.p = None
        self.pdr = None
        self.use_intensities = True
        self.use_matched_lines = True

        # Tracks states and history data
        self.remove_line_idxs = [[], []]
        self.new_matched_pair = [None, None]
        self.matched_lines = []
        self.action_history = []

    # Determines the widget layout
    def make_box_layout(self):
        return widgets.Layout(
            display="grid",
            flex_flow="column",
            align_items="stretch",
            border="solid 1px black",
            margin="5px",
            padding="5px",
        )

    # Activates an initializes widget
    def activate_widget(self):

        IPython.get_ipython().run_line_magic("matplotlib", "widget")

        if self.active:
            print("Widget is already activated.")
            return
        self.active = True

        output_plot = widgets.Output()
        with output_plot:
            self.fig, self.ax = plt.subplots(
                nrows=5,
                ncols=1,
                constrained_layout=True,
                figsize=(10, 8),
                gridspec_kw={"height_ratios": [1, 1, 2, 1, 1]},
            )

        self.initial_draw()
        self.fig.canvas.toolbar_position = "bottom"
        self.fig.canvas.header_visible = False

        # Initializes interactive components of the widget
        reset_top_plot = widgets.Button(description="Reset Upper Lines")
        reset_bottom_plot = widgets.Button(description="Reset Lower Lines")
        toggle_intensity = widgets.ToggleButton(
            value=True, description="Toggle Intensity"
        )
        toggle_fit = widgets.ToggleButton(value=True, description="Toggle Matched Fit")
        residual_slider = widgets.FloatSlider(
            description="Residual Cap", min=1, max=30, value=self.res_size
        )
        top_slider = widgets.FloatSlider(
            description="Upper Size", min=0.1, max=10, value=self.gauss_stds[0]
        )
        bottom_slider = widgets.FloatSlider(
            description="Lower Size", min=0.1, max=10, value=self.gauss_stds[1]
        )
        fit_order_slider = widgets.SelectionSlider(
            options=range(1, 10), value=self.fit_order, description="Fit Order"
        )

        self.cursor_mode = widgets.Button(
            description="Remove", style={"button_color": "lightgreen"}
        )
        close_widget = widgets.Button(description="Close Widget", button_style="danger")
        undo_button = widgets.Button(description="Undo")

        # Groups widget controls into similar types
        controls_left = widgets.VBox(
            [reset_top_plot, reset_bottom_plot, top_slider, bottom_slider],
            layout=self.make_box_layout(),
        )
        controls_center = widgets.VBox(
            [
                toggle_intensity,
                toggle_fit,
                undo_button,
                residual_slider,
                fit_order_slider,
            ],
            layout=self.make_box_layout(),
        )
        controls_right = widgets.VBox(
            [self.cursor_mode, close_widget], layout=self.make_box_layout()
        )
        main_layout = widgets.HBox(
            [output_plot, controls_left, controls_center, controls_right]
        )
        self.children = [main_layout]

        # Tells each widget what to do when they are interacted with
        reset_top_plot.on_click(lambda event: self.reset_line_data(event, 0))
        reset_bottom_plot.on_click(lambda event: self.reset_line_data(event, 1))
        top_slider.observe(lambda ch: self.adjust_line_std(0, ch.new), names="value")
        bottom_slider.observe(lambda ch: self.adjust_line_std(1, ch.new), names="value")
        toggle_intensity.observe(
            lambda _: self.change_intensity_setting(), names="value"
        )
        toggle_fit.observe(lambda _: self.change_fit_setting(), names="value")
        self.cursor_mode.on_click(self.change_mode)
        undo_button.on_click(self.undo_action)
        residual_slider.observe(
            lambda ch: self.adjust_residual_cap(ch.new), names="value"
        )
        fit_order_slider.observe(
            lambda ch: self.adjust_fit_order(ch.new), names="value"
        )
        close_widget.on_click(self.close_widget)

        self.fig.canvas.mpl_connect("button_press_event", self.on_click)
        display(self)

    def change_intensity_setting(self):
        self.use_intensities = not self.use_intensities
        self.update_line_image(0)
        self.update_line_image(1)

    def change_fit_setting(self):
        self.use_matched_lines = not self.use_matched_lines
        self.update_fit_plots()

    def adjust_line_std(self, idx, value):
        self.gauss_stds[idx] = value
        self.update_line_image(idx)

    def adjust_residual_cap(self, value):
        self.res_size = value
        self.update_fit_plots()

    def adjust_fit_order(self, value):
        self.fit_order = value
        if self.valid_fit_possible():
            self.generate_wavelength_calibration()
        self.update_fit_plots()

    def change_mode(self, _):
        colors = ["lightgreen", "lightblue"]
        modes = ["Remove", "Match"]
        self.mode_idx = (self.mode_idx + 1) % len(modes)
        self.cursor_mode.style.button_color = colors[self.mode_idx]
        self.cursor_mode.description = modes[self.mode_idx]
        self.interaction = modes[self.mode_idx]

    def reset_line_data(self, event=None, ax_idx=None):
        if ax_idx is None:
            return

        self.remove_line_idxs[ax_idx] = []
        self.new_matched_pair = [None, None]
        self.matched_lines = []

        # Redraws images
        self.update_line_image(0)
        self.update_line_image(1)
        self.draw_matched_line_data()

        # Updates wavelength calibration
        if self.valid_fit_possible():
            self.generate_wavelength_calibration()
        else:
            self.update_fit_plots()

    def close_widget(self, _):
        if not self.active:
            print("Widget is not activated yet.")
            return
        IPython.get_ipython().run_line_magic("matplotlib", "inline")
        self.final_lines = [self.filter_out_lines(0)[0], self.filter_out_lines(1)[0]]
        self.children = []
        plt.close(self.fig)
        self.active = False

    def undo_action(self, _):
        if not self.action_history:
            return
        last = self.action_history.pop()
        if last["action"] == "Remove":
            self.remove_line_idxs[last["plot_idx"]].remove(last["line_idx"])
            self.update_line_image(last["plot_idx"])
            self.update_fit_plots()
        elif last["action"] == "Match":
            if last["pair"] in self.matched_lines:
                self.matched_lines.remove(last["pair"])
                self.draw_matched_line_data()
                self.update_fit_plots()

    def filter_out_lines(self, idx):
        keep_mask = ~np.isin(
            np.arange(len(self.original_lines[idx])), self.remove_line_idxs[idx]
        )
        return (
            self.original_lines[idx][keep_mask],
            self.original_intensities[idx][keep_mask],
        )

    def valid_fit_possible(self):

        # Determines whether enough lines are matched for a valid fit
        if self.use_matched_lines:
            return len(self.matched_lines) >= self.fit_order + 1
        else:
            l0 = self.filter_out_lines(0)[0]
            l1 = self.filter_out_lines(1)[0]
            return len(l0) > 0 and len(l0) == len(l1)

    def update_line_image(self, idx):
        lines, intensities = self.filter_out_lines(idx)
        if not self.use_intensities:
            intensities = np.ones_like(intensities)
        std = self.gauss_stds[idx]
        x_range = max(lines) - min(lines)
        xs = np.linspace(min(lines) - 0.05 * x_range, max(lines) + 0.05 * x_range, 1000)
        ys = np.sum(
            intensities[:, None]
            * np.exp(-((xs[None, :] - lines[:, None]) ** 2 / (2 * std**2))),
            axis=0,
        )

        if self.images[idx] is None:
            self.images[idx] = self.ax[idx].imshow(
                ys[np.newaxis, :],
                aspect="auto",
                cmap="Greys_r",
                extent=[xs[0], xs[-1], 0, 1],
                vmin=0,
                vmax=1,
            )
        else:
            self.images[idx].set_array(ys[np.newaxis, :])
            self.images[idx].set_extent([xs[0], xs[-1], 0, 1])
        self.ax[idx].set_ylabel(f"{len(lines)} lines")
        self.fig.canvas.draw_idle()

    def update_fit_plots(self):
        for ax in self.ax[2:]:
            ax.clear()
        if not self.valid_fit_possible() or self.p is None:
            return
        if self.use_matched_lines:
            x_data, y_data = zip(*self.matched_lines)
        else:
            x_data = self.filter_out_lines(0)[0]
            y_data = self.filter_out_lines(1)[0]
        xs = np.linspace(min(x_data), max(x_data), 10000)
        self.ax[2].scatter(x_data, y_data, color="k")
        self.ax[2].plot(xs, self.p(xs), "k--")
        slopes = np.diff(y_data) / np.diff(x_data)
        avg_pos = (np.array(x_data[:-1]) + np.array(x_data[1:])) / 2
        self.ax[3].plot(xs, self.pdr(xs), "k--")
        self.ax[3].scatter(avg_pos, slopes, color="k")
        residuals = np.array(y_data) - self.p(x_data)
        self.ax[4].stem(x_data, residuals, markerfmt="kx", basefmt=" ", linefmt="k")
        self.ax[4].axhline(0, color="black", ls="dotted")
        self.ax[4].set_ylim(-self.res_size, self.res_size)

    def generate_wavelength_calibration(self):
        if self.use_matched_lines:
            x_data, y_data = zip(*self.matched_lines)
        else:
            x_data = self.filter_out_lines(0)[0]
            y_data = self.filter_out_lines(1)[0]
        z = np.polyfit(x_data, y_data, self.fit_order)
        self.p = np.poly1d(z)
        self.pdr = np.polyder(self.p, 1)
        self.update_fit_plots()

    def initial_draw(self):
        for idx in range(2):
            self.update_line_image(idx)
        if self.valid_fit_possible():
            self.generate_wavelength_calibration()

    def on_click(self, event):

        # Determine which plot was clicked
        idxs = np.where(self.ax == event.inaxes)[0]
        if not len(idxs):
            return
        ax_idx = idxs[0]

        # Handles removing logic
        if self.interaction == "Remove" and ax_idx in [0, 1]:

            # Prevents running if there are no lines left
            available = [
                i
                for i in range(len(self.original_lines[ax_idx]))
                if i not in self.remove_line_idxs[ax_idx]
            ]
            if not available:
                return

            # Determines which line is closest to the click
            m_idx = min(
                available,
                key=lambda i: abs(self.original_lines[ax_idx][i] - event.xdata),
            )
            self.remove_line_idxs[ax_idx].append(m_idx)
            self.action_history.append(
                {"action": "Remove", "plot_idx": ax_idx, "line_idx": m_idx}
            )

            # Updates visuals
            self.update_line_image(ax_idx)
            if self.valid_fit_possible():
                self.generate_wavelength_calibration()
            self.update_fit_plots()

            # Unmatches line if it was paired up
            unavailable_lines = [pair[ax_idx] for pair in self.matched_lines]
            if self.original_lines[ax_idx][m_idx] in unavailable_lines:
                self.matched_lines = [
                    pair
                    for pair in self.matched_lines
                    if self.original_lines[ax_idx][m_idx] not in pair
                ]
                self.draw_matched_line_data()

        elif self.interaction == "Match" and ax_idx in [0, 1]:

            visible_lines, _ = self.filter_out_lines(ax_idx)
            unavailable_lines = [pair[ax_idx] for pair in self.matched_lines]
            available_lines = [
                line for line in visible_lines if line not in unavailable_lines
            ]

            if not available_lines:
                return

            # Pick the closest visible, unmatched line
            closest_line = min(available_lines, key=lambda i: abs(i - event.xdata))
            self.new_matched_pair[ax_idx] = closest_line

            # When both have been chosen, confirm the match
            if all(line is not None for line in self.new_matched_pair):
                self.matched_lines.append(tuple(self.new_matched_pair))
                self.action_history.append(
                    {"action": "Match", "pair": tuple(self.new_matched_pair)}
                )
                self.new_matched_pair = [None, None]

                # Generate calibration immediately if possible
                if self.valid_fit_possible():
                    self.generate_wavelength_calibration()
                    self.update_fit_plots()

            # Always redraw match visuals after click
            self.draw_matched_line_data()

    def draw_matched_line_data(self):

        # Removes matched line indicators
        for ax in self.ax[:2]:
            for child in ax.lines[:]:
                if child.get_label() == "match-line":
                    child.remove()

        # Removes connections between plots
        for artist in list(self.fig.artists):
            if hasattr(artist, "get_label") and (
                artist.get_label() == "connection-line"
            ):
                artist.remove()

        # Adds red lines for in-progress matches
        for ax_idx in range(2):
            if self.new_matched_pair[ax_idx] is not None:
                self.ax[ax_idx].axvline(
                    self.new_matched_pair[ax_idx],
                    color="red",
                    ls="--",
                    label="match-line",
                )

        # Adds green lines for matched line pairs
        for x1, x2 in self.matched_lines:
            self.ax[0].axvline(x1, color="green", ls="--", label="match-line")
            self.ax[1].axvline(x2, color="green", ls="--", label="match-line")

            con = patches.ConnectionPatch(
                xyA=(x1, 0),
                xyB=(x2, 1),
                coordsA="data",
                coordsB="data",
                axesA=self.ax[0],
                axesB=self.ax[1],
                color="green",
                linestyle="--",
                zorder=-1,
                label="connection-line",
            )
            self.fig.add_artist(con)
        self.fig.canvas.draw_idle()
