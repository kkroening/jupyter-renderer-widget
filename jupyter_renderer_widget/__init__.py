from contextlib import closing
from contextlib import contextmanager
from IPython.display import display
from matplotlib import pyplot as plt
from tqdm.notebook import tqdm
import ffmpeg
import io
import IPython.display
import ipywidgets
import numpy as np
import PIL


DEFAULT_WIDTH = 960
DEFAULT_HEIGHT = 540
DEFAULT_DPI = 96


@contextmanager
def video_pipe_context(filename, width, height):
    process = (
        ffmpeg.input(
            'pipe:', format='rawvideo', pix_fmt='rgb24', s='{}x{}'.format(width, height)
        )
        .output(filename, pix_fmt='yuv420p')
        .overwrite_output()
        .run_async(pipe_stdin=True)
    )
    with closing(process.stdin) as pipe:
        yield pipe
    process.wait()


def render_video(out_filename, render_func, frame_count, tqdm=None):
    def check(image):
        if image.dtype != np.uint8:
            raise TypeError(
                'image dtype must be {}; got {}'.format(np.dtype(np.uint8), image.dtype)
            )
        if len(image.shape) != 3 or image.shape[2] != 3:
            raise ValueError('image shape must be (:,:,3); got {}'.format(image.shape))
        return image

    first_frame = check(render_func(0))
    height, width = first_frame.shape[:2]
    frame_nums = range(frame_count)
    if tqdm is not None:
        frame_nums = tqdm(frame_nums)
    with video_pipe_context(out_filename, width, height) as pipe:
        for frame_num in frame_nums:
            if frame_num == 0:
                frame = first_frame
            else:
                frame = check(render_func(frame_num))
                if frame.shape != first_frame.shape:
                    raise ValueError(
                        'image shape changed from {} to {}'.format(
                            first_frame.shape, frame.shape
                        )
                    )
            pipe.write(frame.tobytes())


def save_pyplot_fig_as_numpy(fig, dpi=DEFAULT_DPI):
    # matplotlib.rcParams['savefig.pad_inches'] = 0
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=dpi)
    buf.seek(0)
    pil_image = PIL.Image.open(buf).convert('RGB')
    return np.array(pil_image)


def render_pyplot_video(
    out_filename,
    render_func,
    frame_count,
    tqdm=None,
    width=960,
    height=540,
    dpi=DEFAULT_DPI,
):
    # old_backend = plt.get_backend()
    # plt.switch_backend('agg')
    try:
        figsize = (int(width / dpi), int(height / dpi))
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        fig.tight_layout()

        def do_render(frame_num):
            ax.clear()
            render_func(ax, frame_num)
            return save_pyplot_fig_as_numpy(fig, dpi=dpi)

        render_video(out_filename, do_render, frame_count, tqdm)
    finally:
        # plt.switch_backend(old_backend)
        fig.clear()


class AutoplayVideo(IPython.display.Video):
    def __init__(
        self,
        data=None,
        url=None,
        filename=None,
        embed=False,
        mimetype=None,
        width=None,
        height=None,
        controls=True,
        autoplay=True,
        loop=True,
    ):
        """An ipywidgets :obj:`Video` that automatically starts playing when displayed."""
        super(AutoplayVideo, self).__init__(
            data, url, filename, embed, mimetype, width, height
        )
        self.controls = controls
        self.autoplay = autoplay
        self.loop = loop

    def _repr_html_(self):
        assert not self.embed, 'Embedding not implemented (yet)'
        options = []
        if self.width:
            options.append('width={}'.format(self.width))
        if self.height:
            options.append('height={}'.format(self.height))
        if self.autoplay:
            options.append('autoplay')
        if self.controls:
            options.append('controls')
        if self.loop:
            options.append('loop')
        url = self.url if self.url is not None else self.filename
        disclaimer = 'Your browser does not support the <code>video</code> element.'
        return '<video src="{}" {}>{}</video>'.format(
            url, ' '.join(options), disclaimer
        )


class Renderer(ipywidgets.VBox):
    def __init__(
        self, render_func, frame_count, out_filename='out.mp4', width=None, height=None
    ):
        render_button = ipywidgets.Button(description='Render')
        render_button.on_click(self._on_render)
        out = ipywidgets.Output()
        super(Renderer, self).__init__([render_button, out])
        self.render_func = render_func
        self.frame_count = frame_count
        self.out_filename = out_filename
        self.width = width
        self.height = height
        self.out = out

    def _render(self):
        render_video(self.out_filename, self.render_func, self.frame_count, tqdm=tqdm)

    def _on_render(self, event):
        with self.out:
            self.out.clear_output()
            self._render()
            display(AutoplayVideo(self.out_filename, width=self.width))


class PyplotRenderer(Renderer):
    def __init__(
        self,
        render_func,
        frame_count,
        out_filename='out.mp4',
        width=None,
        height=None,
        dpi=DEFAULT_DPI,
    ):
        super(PyplotRenderer, self).__init__(
            render_func, frame_count, out_filename, width, height
        )
        self.dpi = dpi

    def _render(self):
        render_pyplot_video(
            self.out_filename,
            self.render_func,
            self.frame_count,
            tqdm=tqdm,
            dpi=self.dpi,
        )
