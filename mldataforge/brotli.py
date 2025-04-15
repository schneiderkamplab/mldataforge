import brotli
import io

__all__ = ["brotli_open"]

def brotli_open(filename, mode='rb', encoding='utf-8'):
    return BrotliFile(filename, mode=mode, encoding=encoding)

class BrotliFile:
    def __init__(self, filename, mode='rb', encoding='utf-8', compress_level=11):
        self.filename = filename
        self.mode = mode
        self.encoding = encoding
        self.compress_level = compress_level

        if 'b' not in mode:
            self.binary = False
            mode = mode.replace('t', 'b')
        else:
            self.binary = True

        self.file = open(filename, mode)

        if 'r' in mode:
            self.decompressor = brotli.Decompressor()
            self._stream = self._make_reader()
        elif 'w' in mode:
            self.compressor = brotli.Compressor(quality=compress_level)
            self._stream = self._make_writer()
        else:
            raise ValueError("Unsupported mode (use 'r' or 'w' with 'b' or 't')")

    def _make_reader(self):
        buffer = io.BytesIO()

        def generator():
            while True:
                chunk = self.file.read(8192)
                if not chunk:
                    break
                decompressed = self.decompressor.decompress(chunk)
                buffer.write(decompressed)
            buffer.seek(0)
            return buffer

        byte_stream = generator()
        return byte_stream if self.binary else io.TextIOWrapper(byte_stream, encoding=self.encoding)

    def _make_writer(self):
        writer = self if self.binary else io.TextIOWrapper(self, encoding=self.encoding)
        return writer

    def write(self, data):
        if isinstance(data, str):
            data = data.encode(self.encoding)
        compressed = self.compressor.process(data)
        self.file.write(compressed)

    def flush(self):
        if hasattr(self, 'compressor'):
            self.file.write(self.compressor.finish())
        self.file.flush()

    def read(self, size=-1):
        return self._stream.read(size)

    def readline(self, size=-1):
        return self._stream.readline(size)

    def __iter__(self):
        return iter(self._stream)

    def close(self):
        if hasattr(self, 'compressor'):
            self.flush()
        self.file.close()

    def __enter__(self):
        return self._stream

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
