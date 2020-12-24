from utils import plots
import pytest


class TestPlots:

    @pytest.mark.xfail(raises=IndexError)    
    def test_activations(self):
        images = ''
        targets = ''
        paths = ''
        fname = ''
        names = ''
        max_size = ''
        max_subplots = ''

        plots.plot_images(images, targets, paths, fname, names, 
            max_size, max_subplots)