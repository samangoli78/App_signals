"""STFT / energy-overlay parameters for signal plots.

Values are single-element lists so tk sliders and spinboxes can mutate them in place.
"""


class SpectrogramSettings:
    def __init__(self) -> None:
        self.n_fft = [100]
        self.hop_length = [5]
        self.win_length = [35]
        self.high_b0 = [40]
        self.high_b1 = [200]
        self.low_b0 = [3]
        self.low_b1 = [150]
        self.len_hann = [5]
        self.max_pooling_length = [1]
        self.TH = [0.45]
