# taskMap.py - builds a PNG and interactive HTML map from the task export: one pin
# per open task, placed at its street address and colored by status.

from maybemap.config import EXPORT_INTERACTIVE_HTML, TASK_DATA_PATH
from maybemap.interactive import export_geojson, export_html
from maybemap.render import render_png
from maybemap.sources.tasks import load_tasks


def main():
    data = load_tasks(TASK_DATA_PATH)
    render_png(data)
    export_geojson(data)
    if EXPORT_INTERACTIVE_HTML:
        export_html(data)


if __name__ == "__main__":
    main()
