import struct

__all__ = ["decode_a85_stream_to_file"]

def decode_a85_stream_to_file(base85_string, output_file_path):
    packI = struct.Struct('!I').pack

    pending = []
    with open(output_file_path, "wb") as f_out:
        for c in base85_string:
            if c.isspace():
                continue
            if c == 'z':
                if pending:
                    raise ValueError("Invalid 'z' inside base85 tuple")
                f_out.write(b'\x00\x00\x00\x00')
            else:
                pending.append(ord(c) - 33)
                if len(pending) == 5:
                    acc = 0
                    for x in pending:
                        acc = acc * 85 + x
                    f_out.write(packI(acc))
                    pending.clear()

        # Carefully handle pending tail
        if pending:
            missing = 5 - len(pending)
            for _ in range(missing):
                pending.append(84)  # pad with 'u' (which is 84)

            acc = 0
            for x in pending:
                acc = acc * 85 + x
            decoded = packI(acc)
            f_out.write(decoded[:4 - missing])
