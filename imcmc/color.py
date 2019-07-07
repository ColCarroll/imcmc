import itertools

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import scipy.stats as st


class ImageLines:
    def __init__(self, image, strategy):
        self.image = image
        self.strategy = strategy

    def make_segments(self, start, end):
        line = self.make_line(start, end)
        x, y = line.astype(int).T
        colors = self.image[x, y, :]
        if colors.dtype == np.uint8:
            colors = colors / 256.0
        colors = np.concatenate((colors, 0.9 * np.ones((colors.shape[0], 1))), axis=1)[
            :-1
        ]
        line = line[:, -1::-1]
        line[:, 1] = self.image.shape[0] - line[:, 1]
        points = line.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        return segments, colors

    def make_line(self, start, end):
        """Create a line from `start` to `end`, with points at all integer coordinates on the way.

        The strategy is to find all the integer coordinates in the x and y
        coordinates separately, then merge them with a `sort`.
        """
        grad = (end - start).reshape(-1, 1)
        t = np.sort(
            np.hstack(
                (
                    np.linspace(0, 1, abs(grad[0, 0]) + 1, endpoint=True),
                    np.linspace(0, 1, abs(grad[1, 0]), endpoint=False)[1:],
                )
            )
        )
        return np.dot(grad, t[None, :]).T + start

    def plot(self, n_points=1000, linewidth=2, ax=None):
        if ax is None:
            fig, ax = plt.subplots(
                figsize=(10, 10 * self.image.shape[0] / self.image.shape[1])
            )

        segments, colors = zip(
            *[
                self.make_segments(*p)
                for p in self.strategy.gen_points(self.image, n_points)
            ]
        )
        lines = LineCollection(
            np.vstack(segments), colors=np.vstack(colors), linewidths=linewidth
        )

        ax.add_collection(lines)
        ax.set_xlim(0, self.image.shape[1])
        ax.set_ylim(0, self.image.shape[0])

        ax.yaxis.set_visible(False)
        ax.xaxis.set_visible(False)

        for spine in ax.spines.values():
            spine.set_visible(False)
        return fig, ax


class UniformPathStrategy:
    def gen_points(self, image, n_points):
        end = np.array(
            [np.random.randint(0, image.shape[0]), np.random.randint(0, image.shape[1])]
        )
        start = None
        points = []
        for _ in range(n_points):
            start, end = (
                end,
                np.array(
                    [
                        np.random.randint(0, image.shape[0]),
                        np.random.randint(0, image.shape[1]),
                    ]
                ),
            )
            points.append((start, end))
        return points


class UniformStrategy:
    def gen_points(self, image, n_points):
        points = [
            (
                np.array(
                    [
                        np.random.randint(0, image.shape[0]),
                        np.random.randint(0, image.shape[1]),
                    ]
                ),
                np.array(
                    [
                        np.random.randint(0, image.shape[0]),
                        np.random.randint(0, image.shape[1]),
                    ]
                ),
            )
            for _ in range(n_points)
        ]
        return points


class GibbsIntensityStrategy:
    def __init__(self, dark=True):
        self.dark = dark

    def gen_points(self, image, n_points):
        end = np.array(
            [np.random.randint(0, image.shape[0]), np.random.randint(0, image.shape[1])]
        )
        start = None
        points = []
        pdf = image.sum(axis=-1)
        pdf = pdf / image.sum()
        pdf = pdf * pdf
        if self.dark:
            pdf = 1 - pdf
        col_pdf = pdf / pdf.sum(axis=0)
        row_pdf = (pdf.T / pdf.sum(axis=1)).T
        for idx in range(n_points):
            start = end.copy()
            if idx % 2:
                end[1] = np.random.choice(
                    np.arange(image.shape[1]), p=row_pdf[end[0], :]
                )
            else:
                end[0] = np.random.choice(
                    np.arange(image.shape[0]), p=col_pdf[:, end[1]]
                )
            points.append((start.copy(), end.copy()))
        return points


class GibbsUniformStrategy:
    def gen_points(self, image, n_points):
        end = np.array(
            [np.random.randint(0, image.shape[0]), np.random.randint(0, image.shape[1])]
        )
        start = None
        points = []
        for idx in range(n_points):
            start = end.copy()
            end[idx % 2] = np.random.randint(0, image.shape[idx % 2])
            points.append((start.copy(), end.copy()))
        return points


class UniformLinesStrategy:
    def gen_points(self, image, n_points):
        height, width = image.shape[:2]
        horiz = np.random.binomial(n_points, height / (height + width))
        vert = n_points - horiz

        h_lines = np.random.randint(0, height, size=horiz)
        xvals = np.random.randint(0, width, size=(horiz, 2))
        v_lines = np.random.randint(0, width, size=vert)
        yvals = np.random.randint(0, height, size=(vert, 2))

        points = []
        for ((x1, x2), y) in zip(xvals, h_lines):
            points.append((np.array([y, x1]), np.array([y, x2])))
        for (x, (y1, y2)) in zip(v_lines, yvals):
            points.append((np.array([y1, x]), np.array([y2, x])))
        return points


class IntensityMCMCStrategy:
    def __init__(self, step_size=None, dark=True):
        self.step_size = step_size
        self.dark = dark

    def image_mcmc(self, image):
        if self.step_size is None:
            step_size = min(image.shape[:2]) ** 2 // 50
        else:
            step_size = self.step_size
        pdf = image.sum(axis=-1)
        pdf = pdf / image.sum()
        pdf = pdf * pdf
        if self.dark:
            pdf = 1 - pdf
        log_pdf = np.log(pdf) - np.log(pdf.sum())
        ylim, xlim = pdf.shape

        proposal = st.multivariate_normal(
            cov=step_size * np.diag(pdf.shape[-1::-1]) / min(pdf.shape)
        )

        current = (np.random.randint(0, ylim), np.random.randint(0, xlim))
        while True:
            jump = proposal.rvs().astype(int)
            prop = tuple(current + jump)
            if any(p < 0 for p in prop) or prop[0] >= ylim or prop[1] >= xlim:
                continue
            elif np.log(np.random.rand()) < log_pdf[prop] - log_pdf[current]:
                yield np.array(current), np.array(prop)
                current = prop

    def gen_points(self, image, n_points):
        return list(itertools.islice(self.image_mcmc(image), n_points))


class RandomWalkStrategy:
    def __init__(self, scale=15):
        self.scale = scale

    def gen_points(self, image, n_points):
        start = np.array(
            [np.random.randint(0, image.shape[0]), np.random.randint(0, image.shape[1])]
        )
        points = start + np.cumsum(
            np.random.randint(0, 2 * self.scale + 1, size=(n_points + 1, 2))
            - self.scale,
            axis=0,
        )

        for idx in (0, 1):
            points[:, idx] = np.abs(points[:, idx])
            points[:, idx] = np.mod(points[:, idx], 2 * image.shape[idx])
            points[:, idx] = (
                image.shape[idx] - 1 - np.abs(points[:, idx] - image.shape[idx] - 1)
            )

        return list(zip(points[:-1], points[1:]))
