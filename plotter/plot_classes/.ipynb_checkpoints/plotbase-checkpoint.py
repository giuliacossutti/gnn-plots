from plotter.config_dict import ConfigDict


class PlotBase:
    def __init__(self, **kwargs):
        # use ConfigDict to store kwargs
        self.config = ConfigDict(**kwargs)

    def plot(self):
        raise NotImplementedError("Subclasses should implement this method")
