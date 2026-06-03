"""Generate Week 1 AI TTS MP3 files for the Life Science spelling quiz.

Usage:
  pip install openai
  export OPENAI_API_KEY="your_api_key_here"
  python scripts/generate_week1_audio.py

Output:
  audio/us/week1/<slug>-word.mp3
  audio/us/week1/<slug>-reading.mp3

Notes:
  - The web app will play these MP3 files when present.
  - Missing files safely fall back to browser speech synthesis.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List

from openai import OpenAI

MODEL = "gpt-4o-mini-tts"
VOICE = "marin"
OUTPUT_DIR = Path(__file__).resolve().parents[1] / "audio" / "us" / "week1"

WORDS: List[Dict[str, str]] = [
    {"word": "observe", "meaning": "To look carefully and notice details.", "example": "Scientists observe plants as they grow."},
    {"word": "question", "meaning": "Something you ask because you want to know more.", "example": "A good question can start an experiment."},
    {"word": "predict", "meaning": "To say what you think will happen next.", "example": "I predict the seed will sprout this week."},
    {"word": "experiment", "meaning": "A test used to learn or prove something.", "example": "We did an experiment with water and seeds."},
    {"word": "investigate", "meaning": "To study something carefully to find answers.", "example": "We investigate why some plants grow faster."},
    {"word": "record", "meaning": "To write down information so you can use it later.", "example": "Scientists record their observations in a notebook."},
    {"word": "measure", "meaning": "To find the size, length, amount, or weight of something.", "example": "Measure the leaf with a ruler."},
    {"word": "compare", "meaning": "To look at things and find how they are alike or different.", "example": "Compare the two insects carefully."},
    {"word": "describe", "meaning": "To tell what something is like.", "example": "Describe the color and shape of the flower."},
    {"word": "sort", "meaning": "To put things into groups.", "example": "Sort the animals by their body coverings."},
    {"word": "test", "meaning": "To try something to learn if it works or is true.", "example": "We test which soil helps seeds grow best."},
    {"word": "data", "meaning": "Facts or numbers collected during an investigation.", "example": "Our data shows the plant grew two centimeters."},
    {"word": "results", "meaning": "What you find out after a test or experiment.", "example": "The results showed that sunlight helped the plant."},
    {"word": "evidence", "meaning": "Information that helps show something is true.", "example": "The tracks are evidence that an animal was here."},
    {"word": "tools", "meaning": "Objects used to do a job.", "example": "A ruler and hand lens are science tools."},
    {"word": "scientist", "meaning": "A person who studies the natural world.", "example": "A scientist asks questions and collects evidence."},
    {"word": "notebook", "meaning": "A book used for writing notes or records.", "example": "Write your observations in the notebook."},
    {"word": "model", "meaning": "A simple copy or picture used to explain something.", "example": "We made a model of a plant cell."},
    {"word": "safety", "meaning": "Being protected from danger or harm.", "example": "Wear goggles for safety during the experiment."},
    {"word": "plan", "meaning": "A set of steps for doing something.", "example": "Make a plan before you start the investigation."},
    {"word": "quantitative", "meaning": "Using numbers or measurements.", "example": "Plant height is quantitative data."},
    {"word": "qualitative", "meaning": "Using descriptions instead of numbers.", "example": "Leaf color is qualitative data."},
    {"word": "systematic", "meaning": "Done in an organized step-by-step way.", "example": "A systematic method helps scientists avoid mistakes."},
    {"word": "experimental", "meaning": "Related to a scientific test.", "example": "The experimental group got more sunlight."},
    {"word": "investigation", "meaning": "A careful search for answers.", "example": "The investigation helped us understand the habitat."},
    {"word": "interpretation", "meaning": "An explanation of what information means.", "example": "Her interpretation of the data was clear."},
    {"word": "replication", "meaning": "Repeating a test to see if the same result happens.", "example": "Replication makes science more reliable."},
    {"word": "correlation", "meaning": "A relationship between two things that change together.", "example": "There is a correlation between sunlight and plant growth."},
    {"word": "evaluation", "meaning": "A careful judgment about quality or meaning.", "example": "The evaluation helped us choose the best model."},
    {"word": "methodology", "meaning": "The system of methods used in a study.", "example": "The methodology explains how the experiment was done."},
]


def slugify(word: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", word.lower()).strip("-")


def write_speech(client: OpenAI, text: str, output_path: Path, instructions: str) -> None:
    if output_path.exists():
        print(f"skip existing: {output_path.relative_to(Path.cwd())}")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with client.audio.speech.with_streaming_response.create(
        model=MODEL,
        voice=VOICE,
        input=text,
        instructions=instructions,
    ) as response:
        response.stream_to_file(output_path)
    print(f"wrote: {output_path.relative_to(Path.cwd())}")


def main() -> None:
    client = OpenAI()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    word_instructions = (
        "Speak in clear, natural American English for an elementary school spelling quiz. "
        "Say only the target word once. Use a friendly teacher voice."
    )
    reading_instructions = (
        "Speak in clear, warm American English for an 8-year-old English learner. "
        "Use a friendly teacher voice, natural pacing, and crisp pronunciation."
    )

    for item in WORDS:
        slug = slugify(item["word"])
        word_text = item["word"]
        reading_text = (
            f"The word is {item['word']}. "
            f"{item['word']} means: {item['meaning']} "
            f"Example: {item['example']}"
        )
        write_speech(client, word_text, OUTPUT_DIR / f"{slug}-word.mp3", word_instructions)
        write_speech(client, reading_text, OUTPUT_DIR / f"{slug}-reading.mp3", reading_instructions)


if __name__ == "__main__":
    main()
