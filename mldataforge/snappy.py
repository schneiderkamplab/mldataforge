import snappy
import struct
import io

__all__ = ["snappy_open"]

_CHUNK_SIZE = 8192  # default read block size

def snappy_open(filename, mode='rb', encoding='utf-8'):
    return SnappyFile(filename, mode=mode, encoding=encoding)

class _SnappyWriteWrapper(io.RawIOBase):
    def __init__(self, fileobj):
        self.fileobj = fileobj
        self.buffer = io.BytesIO()

    def write(self, b):
        if not isinstance(b, (bytes, bytearray)):
            raise TypeError("Expected bytes")
        self.buffer.write(b)
        return len(b)

    def flush(self):
        data = self.buffer.getvalue()
        if data:
            compressed = snappy.compress(data)
            length = struct.pack(">I", len(compressed))
            self.fileobj.write(length + compressed)
            self.buffer = io.BytesIO()
        self.fileobj.flush()

    def close(self):
        self.flush()
        self.fileobj.close()

    def writable(self):
        return True


class _SnappyReadWrapper(io.RawIOBase):
    def __init__(self, fileobj):
        self.fileobj = fileobj
        self.buffer = io.BytesIO()
        self.eof = False

    def _fill_buffer(self):
        if self.eof:
            return

        while True:
            length_bytes = self.fileobj.read(4)
            if not length_bytes:
                self.eof = True
                break

            if len(length_bytes) < 4:
                raise IOError("Corrupted stream: incomplete chunk length")

            length = struct.unpack(">I", length_bytes)[0]
            compressed = self.fileobj.read(length)
            if len(compressed) < length:
                raise IOError("Corrupted stream: incomplete chunk data")

            decompressed = snappy.decompress(compressed)
            self.buffer.seek(0, io.SEEK_END)
            self.buffer.write(decompressed)

    def read(self, size=-1):
        self._fill_buffer()
        self.buffer.seek(0)

        if size < 0:
            data = self.buffer.read()
            self.buffer = io.BytesIO()
            return data

        data = self.buffer.read(size)
        rest = self.buffer.read()
        self.buffer = io.BytesIO()
        self.buffer.write(rest)
        return data

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
            self._stream = self._reader() if self.binary else io.TextIOWrapper(self._reader(), encoding=encoding)
        elif 'w' in mode:
            self._stream = self._writer() if self.binary else io.TextIOWrapper(self._writer(), encoding=encoding)
        else:
            raise ValueError("Unsupported mode: use 'rb', 'wb', 'rt', or 'wt'")

    def _reader(self):
        return _SnappyReadWrapper(self.fileobj)

    def _writer(self):
        return _SnappyWriteWrapper(self.fileobj)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        if hasattr(self._stream, 'flush'):
            self._stream.flush()
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

    def tell(self):
        return self._stream.tell()

    def seek(self, offset, whence=io.SEEK_SET):
        return self._stream.seek(offset, whence)

    def readable(self):
        return hasattr(self._stream, "read")

    def writable(self):
        return hasattr(self._stream, "write")

    def seekable(self):
        return hasattr(self._stream, "seek")

    def __iter__(self):
        return iter(self._stream)
