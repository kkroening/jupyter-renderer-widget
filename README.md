# jupyter-renderer-widget

A drop-in `ipywidgets` "Render" button to simplify generation of videos within a Jupyter notebook.

## Overview

When working with Jupyter notebooks, I got fed up with how annoyingly cumbersome it is to turn visualizations into plain-old video files (such as MP4), and kept having to write the same boilerplate over and over again, or battle Pyplot's confusing `FuncAnimation`, with no reasonable way to preview output videos without reinventing the wheel.

While there are likely other solutions out there with more bells and whistles, I wanted to solve this problem once and for all for my own personal uses, but figure that it can't hurt to put the code out there for others to refer to.

## Usage:

`jupyter-renderer-widget` is primarily meant to be used within a Jupyter notebook and supports rendering both via numpy arrays (e.g. PIL, OpenCV, etc.) and Matplotlib/Pyplot figures.

### Numpy array rendering

#### Grayscale

```python
from jupyter_renderer_widget import Renderer
import numpy as np

def draw_box(im, x, y, width, height):
    im[y:y+height, x:x+width] = 0
    return im

def render(frame_num):
    im = np.ones((400, 640)) * 255
    x = 40 + 20 * (frame_num if frame_num < 20 else 40 - frame_num)
    return draw_box(im, x, 100, 80, 80)

display(Renderer(render, 40))
```

<img src="https://raw.githubusercontent.com/kkroening/jupyter-renderer-widget/master/doc/numpy-grayscale.gif" alt="numpy grayscale rendering" width="75%" />

#### RGB

```python
from jupyter_renderer_widget import Renderer
import numpy as np

def draw_colored_box(im, x, y, width, height, color):
    im[y:y+height, x:x+width, :] = color
    return im

def render(frame_num):
    im = np.ones((400, 640, 3), dtype=np.uint8) * 255
    x = 40 + 20 * (frame_num if frame_num < 20 else 40 - frame_num)
    return draw_colored_box(im, x, 100, 80, 80, [0x53, 0x7a, 0xff])

display(Renderer(render, 40))
```

<img src="https://raw.githubusercontent.com/kkroening/jupyter-renderer-widget/master/doc/numpy-rgb.gif" alt="numpy RGB rendering" width="75%" />

### Matplotlib/Pyplot

`jupyter-renderer-widget` also supports rendering using Pyplot via `jupyter_renderer_widget.PyplotRenderer`:
```python
from jupyter_renderer_widget import PyplotRenderer

def render_pyplot(ax, t):
    ax.plot([0, 10], [5, t if t < 10 else 20 - t])
    ax.set_xlim([0, 10])
    ax.set_ylim([0, 10])

PyplotRenderer(render_pyplot, frame_count=20, width=640, height=400)
```

<img src="https://raw.githubusercontent.com/kkroening/jupyter-renderer-widget/master/doc/pyplot.gif" alt="pyplot rendering" width="75%" />

## Lower-level building blocks

In addition to the above primary use case, several lower-level building blocks are used internally and are exported publicly for a-la-carte usage:

- `video_pipe_context`: context manager for streaming raw image data to a file via ffmpeg over a pipe.

- `to_uint8_rgb`: convert various numpy image array formats into a cannonical 8-bit RGB format.

- `render_video`: call a provided `render_func` for each frame of a video, and stream the results to a video file via ffmpeg pipe.

- `save_pyplot_fig_as_numpy`: convert pyplot figure to a numpy array.

- `render_pyplot_video`: same as `render_video`, but for Pyplot rendering (note: the `render_func` receives an extra `ax` argument in addition to the frame number).

- `AutoplayVideo`: IPython component equivalent to `IPython.display.Video`, except that it automatically starts playing when displayed, with configurable looping behavior and other settings.

- `Renderer`: Numpy-based renderer widget as described above.

- `PyplotRenderer`: Pyplot-based renderer widget as described above.

TODO: provide better documentation.
