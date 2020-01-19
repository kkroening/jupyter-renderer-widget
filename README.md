# jupyter-renderer-widget

A drop-in `ipywidgets` "Render" button to simplify generation of videos within a Jupyter notebook.

Usage:

```python
import jupyter_render_widget
from IPython.display import display

def render(frame_num):
    im = np.ones((640, 480, 3), dtype=np.uint8) * 255
    x, y = 40 + 10 * frame_num, 100
    width, height = 50, 50
    im[y:y+height, x:x+width, :] = 0
    return im


display(Renderer(render, 20))
```
