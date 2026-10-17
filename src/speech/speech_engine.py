import pyttsx3
import threading


class SpeechEngine:
    def __init__(self):
        self.rate = 160
        self.volume = 1.0
        self.is_speaking = False

    def speak(self, text):
        if not text:
            return

        text = text.strip()

        if not text:
            return

        # Don't start another speech while one is already running
        if self.is_speaking:
            return

        # Run speech in a separate thread
        speech_thread = threading.Thread(
            target=self._speak_worker,
            args=(text,),
            daemon=True
        )

        speech_thread.start()

    def _speak_worker(self, text):
        self.is_speaking = True

        try:
            engine = pyttsx3.init()

            engine.setProperty("rate", self.rate)
            engine.setProperty("volume", self.volume)

            engine.say(text)
            engine.runAndWait()

            engine.stop()

        except Exception as e:
            print(f"[Speech Error] {e}")

        finally:
            self.is_speaking = False