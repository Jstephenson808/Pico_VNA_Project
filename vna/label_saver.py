import threading
from datetime import datetime

from pynput import keyboard


class LabelMaker:
    def __init__(
        self,
        output_file_path: Path,
    ):
        pass

    def save_timestamp(self):
        now = datetime.now()
        timestamp = now.strftime("%Y-%m-%d %H:%M:%S.%f")
        with open(TIMESTAMPS_FILE_OUTPUT_PATH, "a") as f:
            f.write(timestamp + "\n")
        print(f"Time recorded: {timestamp}")

    def on_press(self, key):
        try:
            if key.char == "t":  # Press 't' to mark the time
                self.save_timestamp()
        except AttributeError:
            pass

    def start_listener(self):
        with keyboard.Listener(on_press=self.on_press) as listener:
            listener.join()

    listener_thread = threading.Thread(target=start_listener, daemon=True)
    listener_thread.start()

    print("Press 't' to record time. Press Ctrl+C to exit.")

    try:
        while True:
            pass  # your main loop here
    except KeyboardInterrupt:
        print("\nProgram exited. Timestamps saved in", TIMESTAMPS_FILE_OUTPUT_PATH)
