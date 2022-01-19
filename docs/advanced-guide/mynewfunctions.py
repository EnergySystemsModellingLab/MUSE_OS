from typing import Any, Text
from muse.outputs.sinks import register_output_sink, sink_to_file

@register_output_sink(name="txt")
@sink_to_file(".txt")
def text_dump(data: Any, filename: Text) -> None:
    from pathlib import Path
    Path(filename).write_text(f"Hello world!\n\n{data}")
