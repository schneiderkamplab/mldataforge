import brotli
import io

__all__ = ["brotli_open"]

def brotli_open(filename, mode='rb', encoding='utf-8', quality=11, lgwin=22, lgblock=0):
    return BrotliFile(filename, mode=mode, encoding=encoding, quality=quality, lgwin=lgwin, lgblock=lgblock)

class BrotliFile:
    def __init__(self, filename, mode='rb', encoding='utf-8', quality=11, lgwin=22, lgblock=0):
        self.filename = filename
        self.encoding = encoding

        self.binary = 'b' in mode
        file_mode = mode.replace('t', 'b')
        self.file = open(filename, file_mode)

        if 'r' in mode:
            self._decompressor = brotli.Decompressor()
            self._stream = self._wrap_reader()
        elif 'w' in mode:
            self._compressor = brotli.Compressor(quality=quality, lgwin=lgwin, lgblock=lgblock)
            self._stream = self._wrap_writer()
        else:
            raise ValueError("Unsupported mode (use 'rb', 'wb', 'rt', or 'wt')")

    def _wrap_reader(self):
        buffer = io.BytesIO()
        while True:
            chunk = self.file.read(8192)
            if not chunk:
                break
            buffer.write(self._decompressor.process(chunk))
        buffer.seek(0)
        return buffer if self.binary else io.TextIOWrapper(buffer, encoding=self.encoding)

    def _wrap_writer(self):
        return self if self.binary else io.TextIOWrapper(self, encoding=self.encoding)

    def write(self, data):
        if isinstance(data, str):
            data = data.encode(self.encoding)
        compressed = self._compressor.process(data)
        self.file.write(compressed)
        return len(data)

    def flush(self):
        if hasattr(self, '_compressor'):
            self.file.write(self._compressor.finish())
        self.file.flush()

    def read(self, *args, **kwargs):
        return self._stream.read(*args, **kwargs)

    def readline(self, *args, **kwargs):
        return self._stream.readline(*args, **kwargs)

    def __iter__(self):
        return iter(self._stream)

    def close(self):
        try:
            if hasattr(self._stream, 'flush'):
                self._stream.flush()
        finally:
            self.file.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def tell(self):
        return self._stream.tell()
