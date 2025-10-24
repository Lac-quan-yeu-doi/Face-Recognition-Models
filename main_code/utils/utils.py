import sys

class Tee:
    def __init__(self, *files):
        self.files = files
        self.primary = files[0] if files else sys.stdout
    
    def write(self, text):
        for file in self.files:
            file.write(text)
    
    def flush(self):
        for file in self.files:
            file.flush()
            
    def fileno(self):
        return self.primary.fileno()