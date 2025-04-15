import snappy
import struct
import io

__all__ = ["snappy_open"]

_CHUNK_SIZE = 8192  # 8KB per chunk

def snappy_open(path, mode="rt", encoding=None):
    return SnappyFile(path, mode=mode, encoding=encoding)

class _SnappyWriteWrapper(io.RawIOBase):
    def __init__(self, fileobj):
        self.fileobj = fileobj

    def write(self, b):
        if not isinstance(b, (bytes, bytearray)):
            raise TypeError("Expected bytes")
        compressed = snappy.compress(b)
        length = struct.pack(">I", len(compressed))
        self.fileobj.write(length + compressed)
        return len(b)

    def flush(self):
        self.fileobj.flush()

    def close(self):
        self.fileobj.close()


class _SnappyReadWrapper(io.RawIOBase):
    def __init__(self, fileobj):
        self.fileobj = fileobj
        self.buffer = io.BytesIO()

    def read(self, size=-1):
        result = bytearray()

        while size < 0 or len(result) < size:
            length_bytes = self.fileobj.read(4)
            if not length_bytes:
                break
            if len(length_bytes) < 4:
                raise IOError("Corrupted stream: incomplete chunk length")

            length = struct.unpack(">I", length_bytes)[0]
            compressed = self.fileobj.read(length)
            if len(compressed) < length:
                raise IOError("Corrupted stream: incomplete chunk data")

            result.extend(snappy.decompress(compressed))

        return bytes(result) if size != -1 else result

    def readable(self):
        return True

    def close(self):
        self.fileobj.close()

class SnappyFile:
    def __init__(self, filename, mode='rb', encoding='utf-8'):
        self.filename = filename
        self.mode = mode
        self.encoding = encoding

        self.binary = 'b' in mode
        raw_mode = mode.replace('t', 'b')
        self.fileobj = open(filename, raw_mode)

        if 'r' in mode:
            self._is_reading = True
            self._stream = self._reader() if self.binary else io.TextIOWrapper(self._reader(), encoding=encoding)
        elif 'w' in mode:
            self._is_reading = False
            self._stream = self._writer() if self.binary else io.TextIOWrapper(self._writer(), encoding=encoding)
        else:
            raise ValueError("Mode must include 'r' or 'w'")

    def _reader(self):
        return _SnappyReadWrapper(self.fileobj)

    def _writer(self):
        return _SnappyWriteWrapper(self.fileobj)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def close(self):
        self._stream.close()

    def flush(self):
        if hasattr(self._stream, 'flush'):
            self._stream.flush()

    def read(self, *args, **kwargs):
        return self._stream.read(*args, **kwargs)

    def write(self, *args, **kwargs):
        return self._stream.write(*args, **kwargs)

    def readline(self, *args, **kwargs):
        return self._stream.readline(*args, **kwargs)

    def __iter__(self):
        return iter(self._stream)
