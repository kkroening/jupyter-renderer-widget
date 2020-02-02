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
import PIL.Image


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


def to_uint8_rgb(image):
    if image.dtype != np.uint8:
        # TODO: possibly normalize values.
        image = image.astype(np.uint8)
    if len(image.shape) == 2:
        image = image[:, :, np.newaxis]
        depth = 1
    elif len(image.shape) == 3:
        depth = image.shape[-1]
    else:
        raise ValueError('image shape must be 2D or 3D; got {}'.format(image.shape))
    if depth == 1:
        image = np.tile(image, [1, 1, 3])
    elif depth == 4:
        image = image[:, :, :3]
    elif depth != 3:
        raise ValueError(
            'image depth must be either 1 (grayscale), 3 (RGB), or 4 (RGBA); got {}'.format(
                image.shape
            )
        )
    return image


def render_video(
    out_filename, render_func, frame_count, tqdm=None, start_frame=0, end_frame=-1
):
    first_frame = to_uint8_rgb(render_func(0))
    height, width, depth = first_frame.shape
    if end_frame < 0:
        end_frame = frame_count + end_frame
        assert end_frame >= 0
    frame_nums = range(start_frame, end_frame + 1)
    if tqdm is not None:
        frame_nums = tqdm(frame_nums)
    with video_pipe_context(out_filename, width, height) as pipe:
        for frame_num in frame_nums:
            if frame_num == 0:
                frame = first_frame
            else:
                frame = to_uint8_rgb(render_func(frame_num))
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


def _init_pyplot(width, height, dpi):
    figsize = (int(width / dpi), int(height / dpi))
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    fig.tight_layout()
    return fig, ax


def render_pyplot_video(
    out_filename,
    render_func,
    frame_count,
    tqdm=None,
    width=DEFAULT_WIDTH,
    height=DEFAULT_HEIGHT,
    dpi=DEFAULT_DPI,
):
    # old_backend = plt.get_backend()
    # plt.switch_backend('agg')
    try:
        fig, ax = _init_pyplot(width, height, dpi)

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
        self,
        render_func,
        frame_count,
        out_filename='out.mp4',
        width=None,
        height=None,
        preview=True,
        preview_frame=0,
    ):
        render_button = ipywidgets.Button(description='Render')
        render_button.on_click(self._on_render)
        out = ipywidgets.Output()
        super(Renderer, self).__init__([render_button, out])
        self.render_func = render_func
        self.frame_count = frame_count
        self.preview_frame = min(preview_frame, self.frame_count - 1)
        self.out_filename = out_filename
        self.width = width
        self.height = height
        self.out = out
        self.start_frame = 0
        self.end_frame = frame_count - 1
        if preview:
            with self.out:
                frame_range = (0, frame_count - 1, 1)

                @ipywidgets.interact(
                    preview_frame=frame_range,
                    start_frame=frame_range,
                    end_frame=frame_range,
                )
                def show_preview(
                    preview_frame=self.preview_frame,
                    start_frame=self.start_frame,
                    end_frame=self.end_frame,
                ):
                    self._show_preview(preview_frame)

    def _show_preview(self, preview_frame):
        image = self.render_func(preview_frame)
        rgb_image = to_uint8_rgb(image)
        display(PIL.Image.fromarray(rgb_image, 'RGB'))

    def _render(self):
        render_video(
            self.out_filename,
            self.render_func,
            self.frame_count,
            tqdm=tqdm,
            start_frame=self.start_frame,
            end_frame=self.end_frame,
        )

    def _on_render(self, event):
        with self.out:
            self._render()
            self.out.clear_output()
            display(AutoplayVideo(self.out_filename, width=self.width))


class PyplotRenderer(Renderer):
    def __init__(
        self,
        render_func,
        frame_count,
        out_filename='out.mp4',
        width=DEFAULT_WIDTH,
        height=DEFAULT_HEIGHT,
        dpi=DEFAULT_DPI,
        preview=True,
    ):
        self.dpi = dpi
        super(PyplotRenderer, self).__init__(
            render_func, frame_count, out_filename, width, height, preview
        )

    def _show_preview(self, preview_frame):
        fig, ax = _init_pyplot(self.width, self.height, self.dpi)
        self.render_func(ax, preview_frame)

    def _render(self):
        render_pyplot_video(
            self.out_filename,
            self.render_func,
            self.frame_count,
            width=self.width,
            height=self.height,
            tqdm=tqdm,
            dpi=self.dpi,
        )
