
import grain.python as grain
print(f"grain.ReadOptions defaults: {grain.ReadOptions()}")
try:
    print(f"grain.MultiprocessingOptions defaults: {grain.MultiprocessingOptions()}")
except AttributeError:
    print("grain.MultiprocessingOptions not found")
