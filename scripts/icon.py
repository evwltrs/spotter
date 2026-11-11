# spotter's icon: four corner brackets, #fabd2f on #1d2021, drawn to the pixel at each size rather
# than scaled, so the small ones stay sharp. Writes assets/icon.png (512) and assets/icon.ico.
#   python scripts/icon.py
import os, struct, zlib

BG, FG = (0x1D, 0x20, 0x21, 255), (0xFA, 0xBD, 0x2F, 255)


def draw(s):
    m = round(s * 0.16)              # margin
    a = round(s * 0.26)              # arm
    t = max(1, round(s * 0.085))     # thickness
    px = [[BG] * s for _ in range(s)]
    for (x0, y0, dx, dy) in [(m, m, 1, 1), (s - 1 - m, m, -1, 1), (m, s - 1 - m, 1, -1), (s - 1 - m, s - 1 - m, -1, -1)]:
        for i in range(a):
            for j in range(t):
                px[y0 + j * dy][x0 + i * dx] = FG
                px[y0 + i * dy][x0 + j * dx] = FG
    return px


def png(px):
    s = len(px)
    raw = b''.join(b'\0' + b''.join(bytes(p) for p in row) for row in px)
    chunk = lambda kind, data: struct.pack('>I', len(data)) + kind + data + struct.pack('>I', zlib.crc32(kind + data))
    return b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('>IIBBBBB', s, s, 8, 6, 0, 0, 0)) + chunk(b'IDAT', zlib.compress(raw, 9)) + chunk(b'IEND', b'')


def ico(sizes):
    images = [png(draw(s)) for s in sizes]
    out = struct.pack('<HHH', 0, 1, len(sizes))
    at = 6 + 16 * len(sizes)
    for s, im in zip(sizes, images):
        out += struct.pack('<BBBBHHII', s % 256, s % 256, 0, 0, 1, 32, len(im), at)
        at += len(im)
    return out + b''.join(images)


assets = os.path.join(os.path.dirname(__file__), '..', 'assets')
open(os.path.join(assets, 'icon.png'), 'wb').write(png(draw(512)))
open(os.path.join(assets, 'icon.ico'), 'wb').write(ico([16, 20, 24, 32, 40, 48, 64, 128, 256]))
