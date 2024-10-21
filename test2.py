from pyvirtualdisplay import Display
import time

print("Starting virtual display setup...")

# Start the timer to measure how long the setup takes
start_time = time.time()

try:
    print("Initializing Display object...")
    display = Display(visible=0, size=(1400, 900))
    print("Starting Display...")
    display.start()
    print("Virtual display started.")
except Exception as e:
    print(f"An error occurred: {e}")

# End the timer and print the elapsed time
end_time = time.time()
print(f"Virtual display setup completed in {end_time - start_time:.2f} seconds.")