import queue
import threading
import time
import sounddevice as sd
import sys

# Global variables
audio_queue = queue.Queue()
sample_rate = 44100  # You can adjust this based on your requirements
block_size = 2000  # Number of samples to record at each interval


def audio_callback(indata, frames, time, status):
    """This is called (from a separate thread) for each audio block."""
    if status:
        print(status, file=sys.stderr)
    # Use .copy() to avoid shared memory issues
    locate(indata.copy(), time.inputBufferAdcTime)


def locate(data, time):
    print(time)


def record_audio():
    with sd.InputStream(samplerate=sample_rate, channels=1, callback=audio_callback):
        print("Recording started...")
        # Wait for a short time before starting the first recording
        sd.sleep(500)

        while True:
            try:
                time.sleep(1)
                data, timestamp = audio_queue.get_nowait()
                # Perform musical score alignment here if needed
                print(f"Got audio data at timestamp {timestamp.currentTime}")
                print(data)

            except queue.Empty:
                pass  # Queue is empty, continue looping


# Start the audio recording thread
recording_thread = threading.Thread(target=record_audio, daemon=True)
recording_thread.start()


# Keep the main thread alive
try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("Recording stopped.")
