import os
import tempfile

import imcmc

HERE = os.path.dirname(os.path.abspath(__file__))
EXAMPLE_DIR = os.path.join(os.path.dirname(HERE), "examples")

EXAMPLE = os.path.join(EXAMPLE_DIR, "pymc3-logo.png")


def test_load_gray_image():
    im = imcmc.load_image(EXAMPLE, "L")
    assert im.ndim == 2


def test_load_color_image():
    im = imcmc.load_image(EXAMPLE)
    assert im.ndim == 3


def test_sample_grayscale():
    im = imcmc.load_image(EXAMPLE, "L")
    trace = imcmc.sample_grayscale(im, samples=1000, nchains=4)
    assert len(trace["image"]) == 1000 * 4


def test_plot_multitrace():
    im = imcmc.load_image(EXAMPLE, "L")
    trace = imcmc.sample_grayscale(im, samples=1000, nchains=4)
    imcmc.plot_multitrace(trace, im)


def test_make_gif():
    im = imcmc.load_image(EXAMPLE, "L")
    trace = imcmc.sample_grayscale(im, samples=1000, nchains=4)
    with tempfile.TemporaryDirectory() as tmpdirname:
        filename = os.path.join(tmpdirname, "test_gif.gif")
        assert not os.path.exists(filename)
        imcmc.make_gif(trace, im, steps=2, filename=filename)
        assert os.path.exists(filename)


def test_sample_color():
    im = imcmc.load_image(EXAMPLE)
    trace = imcmc.sample_color(im, samples=1000, nchains=2)
    for color in ("red", "blue", "green"):
        assert len(trace[color]) == 1000 * 2


def test_plot_multitrace_color():
    # just making sure it runs
    im = imcmc.load_image(EXAMPLE)
    trace = imcmc.sample_color(im, samples=1000, nchains=2)
    imcmc.plot_multitrace_color(trace, im)


def test_make_color_gif():
    # just making sure it runs
    im = imcmc.load_image(EXAMPLE)
    trace = imcmc.sample_color(im, samples=1000, nchains=2)
    with tempfile.TemporaryDirectory() as tmpdirname:
        filename = os.path.join(tmpdirname, "test_gif.gif")
        assert not os.path.exists(filename)
        imcmc.make_color_gif(trace, im, steps=2, filename=filename)
        assert os.path.exists(filename)
