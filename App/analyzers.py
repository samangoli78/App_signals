from __future__ import annotations
# Future annotations allow forward-friendly type-hint usage.


class BaseAnalyzer:
    # Polymorphism concept:
    # subclasses share this interface but provide different render implementations.
    def render_signal_axis(self, app, axes, x, bipolar_signal, stimulation_reference):
        # Abstract-style base method:
        # subclasses must implement behavior; base class only defines the interface.
        raise NotImplementedError


class DefaultAnalyzer(BaseAnalyzer):
    # Concrete strategy for normal mode plotting.
    def render_signal_axis(self, app, axes, x, bipolar_signal, stimulation_reference):
        signal_axis = axes["mid"]
        signal_axis.plot(x, bipolar_signal, alpha=0.5, linewidth=0.6)
        signal_axis.set_title(app.cont[app.i][0].loc[app.j, "label_color"])


class TripleExtraAnalyzer(BaseAnalyzer):
    # Alternative strategy delegates to the richer Triple Extra analysis workflow.
    def render_signal_axis(self, app, axes, x, bipolar_signal, stimulation_reference):
        signal_axis = axes["mid"]
        # Method delegation: call back into app for the complex plotting pipeline.
        app.plot_main(signal_axis, x=x, y2=bipolar_signal, arg="mid", reff=stimulation_reference)
