import subprocess

__all__ = ["pigz_open"]

def pigz_open(path, mode="rt", processes=64, encoding=None):
    return PigzFile(path, mode=mode, processes=processes, encoding=encoding)

class PigzFile(object):
    """A wrapper for pigz to handle gzip compression and decompression."""
    def __init__(self, path, mode="rt", processes=4, encoding="utf-8", compression_level=6):
        assert mode in ("rt", "wt", "rb", "wb")
        self.path = path
        self.is_read = mode[0] == "r"
        self.is_text = mode[1] == "t"
        self.processes = processes
        self.encoding = encoding if self.is_text else None
        self._process = None
        self._fw = None
        self.offset = 0
        args = ["pigz", "-p", str(self.processes), "-c"]
        if self.is_read:
            args.extend(("-d", self.path))
            self._process = subprocess.Popen(args, stdout=subprocess.PIPE, encoding=self.encoding, text=self.is_text)
        else:
            args.extend(("-{0}".format(compression_level),))
            self._fw = open(self.path, "w+")
            self._process = subprocess.Popen(args, stdout=self._fw, stdin=subprocess.PIPE, encoding=self.encoding, text=self.is_text)
        
    def __iter__(self):
        assert self.is_read
        for line in self._process.stdout:
            assert isinstance(line, str) if self.is_text else isinstance(line, bytes)
            self.offset += len(line)
            yield line
        self._process.wait()
        assert self._process.returncode == 0
        self._process.stdout.close()
        self._process = None
        
    def write(self, line):
        assert not self.is_read
        assert self._fw is not None
        assert isinstance(line, str) if self.is_text else isinstance(line, bytes)
        self._process.stdin.write(line)
        self.offset += len(line)
    
    def close(self): 
        if self._process:
            if self.is_read:
                self._process.kill()
                self._process.stdout.close()
                self._process = None
            else:
                self._process.stdin.close()
                self._process.wait()
                self._process = None
                self._fw.close()
                self._fw = None

    def tell(self):
        return self.offset

    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
