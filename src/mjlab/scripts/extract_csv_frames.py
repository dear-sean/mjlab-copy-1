"""Extract selected frames from a motion CSV file.

Examples:
  uv run src/mjlab/scripts/extract_csv_frames.py \
    --input-file input.csv --output-file output.csv --frames "[8]"
  uv run src/mjlab/scripts/extract_csv_frames.py \
    --input-file input.csv --output-file output.csv --frames "[1, 8, 20]"
  uv run src/mjlab/scripts/extract_csv_frames.py \
    --input-file input.csv --output-file output.csv \
    --frame-range "(100, 200)"
"""

from __future__ import annotations

import csv
from pathlib import Path

import tyro

import mjlab


def _selected_frame_numbers(
  frames: list[int] | None,
  frame_range: tuple[int, int] | None,
) -> list[int]:
  selected = set(frames or [])

  if frame_range is not None:
    start, end = frame_range
    if start > end:
      raise ValueError(f"Frame range start must not exceed its end: ({start}, {end})")
    selected.update(range(start, end + 1))

  if not selected:
    raise ValueError("Specify --frames, --frame-range, or both")
  if min(selected) < 1:
    raise ValueError("Frame numbers are 1-based and must be positive")

  return sorted(selected)


def main(
  input_file: Path,
  output_file: Path,
  frames: list[int] | None = None,
  frame_range: tuple[int, int] | None = None,
) -> None:
  """Extract frames from a motion CSV into a new CSV file.

  Args:
    input_file: Source motion CSV file.
    output_file: Destination CSV file.
    frames: One or more 1-based frame numbers.
    frame_range: Inclusive 1-based range in the form (start, end).
  """
  if not input_file.is_file():
    raise FileNotFoundError(f"Input file does not exist: {input_file}")
  if input_file.resolve() == output_file.resolve():
    raise ValueError("Input and output files must be different")

  selected = _selected_frame_numbers(frames, frame_range)
  last_requested = selected[-1]
  selected_set = set(selected)
  extracted: list[list[str]] = []
  total_frames = 0

  with input_file.open(newline="", encoding="utf-8-sig") as source:
    reader = csv.reader(source)
    for frame_number, row in enumerate(reader, start=1):
      total_frames = frame_number
      if frame_number in selected_set:
        extracted.append(row)
      if frame_number >= last_requested:
        break

  missing = [frame for frame in selected if frame > total_frames]
  if missing:
    missing_text = ", ".join(map(str, missing))
    raise ValueError(
      f"Requested frames exceed the CSV length ({total_frames}): {missing_text}"
    )

  output_file.parent.mkdir(parents=True, exist_ok=True)
  with output_file.open("w", newline="", encoding="utf-8") as destination:
    csv.writer(destination, lineterminator="\n").writerows(extracted)

  print(
    f"Extracted {len(extracted)} frames from {input_file} to {output_file} "
    f"(source frames: {total_frames})"
  )


if __name__ == "__main__":
  tyro.cli(main, config=mjlab.TYRO_FLAGS)
