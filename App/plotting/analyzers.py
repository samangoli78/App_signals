"""Signal-axis rendering for normal vs triple-extra acquisition modes."""


class DefaultAnalyzer:
    def render_signal_axis(self, app, axes, x, bipolar_signal, stimulation_reference):
        signal_axis = axes["mid"]
        signal_axis.plot(x, bipolar_signal, alpha=0.5, linewidth=0.6)
        signal_axis.set_title(app.cont[app.i][0].loc[app.j, "label_color"])


class TripleExtraAnalyzer:
    def render_signal_axis(self, app, axes, x, bipolar_signal, stimulation_reference):
        signal_axis = axes["mid"]
        app.plot_main(signal_axis, x=x, y2=bipolar_signal, arg="mid", reff=stimulation_reference)
