"""Hot microphone loop using Whisper STT and system text to speech.

This example demonstrates a simple voice interface that listens from the
microphone and replies using the `say` helper from :mod:`lerobot.common.utils`.
Speech transcription relies on the ``speech_recognition`` package which can
use the `whisper` model for offline recognition when installed.

Usage::

    python examples/hotmic_speaklite.py --play-sounds

Press ``Ctrl+C`` to exit the loop.
"""

import argparse

import speech_recognition as sr

from lerobot.common.utils.utils import log_say


def listen_once(play_sounds: bool = True) -> None:
    """Listen from the microphone and speak back the transcript."""
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        log_say("Listening...", play_sounds)
        audio = recognizer.listen(source)
    try:
        text = recognizer.recognize_whisper(audio)
        log_say(f"You said: {text}", play_sounds)
    except sr.UnknownValueError:
        log_say("Could not understand audio.", play_sounds)
    except sr.RequestError as exc:
        log_say(f"STT error: {exc}", play_sounds)


def main() -> None:
    parser = argparse.ArgumentParser(description="Minimal hot microphone example")
    parser.add_argument(
        "--play-sounds",
        action="store_true",
        help="Speak responses using the system text-to-speech",
    )
    args = parser.parse_args()

    log_say("Hotmic ready. Press Ctrl+C to stop.", args.play_sounds)
    try:
        while True:
            listen_once(play_sounds=args.play_sounds)
    except KeyboardInterrupt:
        log_say("Stopping.", args.play_sounds)


if __name__ == "__main__":
    main()

