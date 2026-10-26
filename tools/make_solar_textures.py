"""Gera texturas equirretangulares 2:1 (sem costura nos meridianos/polos) para o
Sol, a Lua e Mercurio, alem dos campos de elevacao (heightmaps) com crateras
parabolicas 3D usados por make_normal_map.py.

A amostragem e feita em coordenadas 3D sobre a esfera unitaria
    p(theta, phi) = (sin(phi)*sin(theta), cos(phi), sin(phi)*cos(theta))
combinando o campo noise.png do framework (amostragem triplanar em varias
oitavas) com bacias de impacto esfericas de perfil parabolico (1 - u^2)^1.6 e
borda elevada. Assim nao ha abertura/costura em theta = 0/2pi nem distorcao
oval nas crateras.
"""
import os
import numpy as np
from PIL import Image


def load_tile_noise(path):
    img = Image.open(path).convert("L")
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return arr


def sample_tile_2d(tile, u, v):
    h, w = tile.shape
    x = np.mod(u * w, w)
    y = np.mod(v * h, h)
    x0 = np.floor(x).astype(np.int32) % w
    y0 = np.floor(y).astype(np.int32) % h
    x1 = (x0 + 1) % w
    y1 = (y0 + 1) % h
    fx = x - np.floor(x)
    fy = y - np.floor(y)
    c00 = tile[y0, x0]
    c10 = tile[y0, x1]
    c01 = tile[y1, x0]
    c11 = tile[y1, x1]
    return (c00 * (1 - fx) + c10 * fx) * (1 - fy) + (c01 * (1 - fx) + c11 * fx) * fy


def triplanar_fbm(tile, px, py, pz, octaves=(1.0, 2.1, 4.3, 8.7, 17.3)):
    # pesos triplanares suaves
    wx = np.abs(px) ** 2 + 1e-4
    wy = np.abs(py) ** 2 + 1e-4
    wz = np.abs(pz) ** 2 + 1e-4
    wsum = wx + wy + wz
    wx /= wsum
    wy /= wsum
    wz /= wsum

    total = np.zeros_like(px, dtype=np.float32)
    amp = 1.0
    amp_sum = 0.0
    for freq in octaves:
        s_yz = sample_tile_2d(tile, (py * 0.5 + 0.5) * freq, (pz * 0.5 + 0.5) * freq)
        s_xz = sample_tile_2d(tile, (px * 0.5 + 0.5) * freq, (pz * 0.5 + 0.5) * freq)
        s_xy = sample_tile_2d(tile, (px * 0.5 + 0.5) * freq, (py * 0.5 + 0.5) * freq)
        val = wx * s_yz + wy * s_xz + wz * s_xy
        total += amp * val
        amp_sum += amp
        amp *= 0.52
    return total / amp_sum


def sphere_grid(width=1024, height=512):
    s = (np.arange(width, dtype=np.float32) + 0.5) / width
    t = (np.arange(height, dtype=np.float32) + 0.5) / height
    theta = s[None, :] * (2.0 * np.pi)
    phi = t[:, None] * np.pi
    sin_phi = np.sin(phi)
    px = sin_phi * np.sin(theta)
    py = np.broadcast_to(np.cos(phi), (height, width)).copy()
    pz = sin_phi * np.cos(theta)
    return px, py, pz


def generate_craters(px, py, pz, seed, specs):
    """Soma bacias e crateras 3D com perfil de tigela parabolica profunda."""
    rng = np.random.default_rng(seed)
    bowl_total = np.zeros_like(px, dtype=np.float32)
    ejecta_total = np.zeros_like(px, dtype=np.float32)

    for count, r_min, r_max, depth_scale in specs:
        # pontos aleatorios uniformes na esfera unitaria
        vec = rng.normal(size=(count, 3)).astype(np.float32)
        vec /= np.linalg.norm(vec, axis=1, keepdims=True)
        radii = rng.uniform(r_min, r_max, size=count).astype(np.float32)
        depths = depth_scale * (radii / r_max) ** 0.75

        for (cx, cy, cz), rad, dep in zip(vec, radii, depths):
            # distancia cordal na esfera: d_chord = |p - c|; para angulos pequenos ~ d_ang
            dot = np.clip(px * cx + py * cy + pz * cz, -1.0, 1.0)
            ang = np.arccos(dot)
            u = ang / rad
            mask = u < 1.45
            if not np.any(mask):
                continue
            um = u[mask]
            # interior da cratera: tigela concava profunda (1 - u^2)^1.5
            inside = np.clip(1.0 - um * um, 0.0, 1.0)
            bowl = -dep * (inside ** 1.45)
            # borda (rim) elevada em torno de u = 1.0
            rim = (0.36 * dep) * np.exp(-((um - 0.98) / 0.15) ** 2)
            # pico central nas crateras maiores
            if rad > 0.14:
                peak = (0.22 * dep) * np.exp(-((um / 0.18) ** 2))
            else:
                peak = 0.0
            bowl_total[mask] += bowl + rim + peak
            ejecta_total[mask] += np.exp(-((um - 0.96) / 0.22) ** 2) * (0.25 * dep)

    return bowl_total, ejecta_total


def make_sun(tile, out_path):
    px, py, pz = sphere_grid(1024, 512)
    gran1 = triplanar_fbm(tile, px, py, pz, octaves=(2.0, 4.5, 9.0, 18.0, 32.0))
    gran2 = triplanar_fbm(tile, -pz, px, py, octaves=(3.0, 7.0, 15.0, 28.0))
    cells = 1.0 - np.abs(2.0 * gran2 - 1.0)  # celulas de conveccao solares
    spots = triplanar_fbm(tile, py, pz, px, octaves=(1.5, 3.2))
    spot_mask = np.clip((spots - 0.66) * 5.0, 0.0, 1.0) ** 2

    plasma = 0.55 * gran1 + 0.45 * cells
    plasma = (plasma - plasma.min()) / (plasma.max() - plasma.min() + 1e-6)
    plasma = np.clip(plasma * (1.0 - 0.35 * spot_mask), 0.0, 1.0)

    # paleta solar continua (laranja-fogo -> ouro -> amarelo brilhante), sem costura
    r = np.clip(0.92 + 0.10 * plasma, 0.0, 1.0)
    g = np.clip(0.46 + 0.46 * (plasma ** 1.15), 0.0, 1.0)
    b = np.clip(0.06 + 0.28 * (plasma ** 2.2), 0.0, 1.0)

    rgb = np.stack([r, g, b], axis=-1)
    Image.fromarray((rgb * 255.0 + 0.5).astype(np.uint8), "RGB").save(out_path, quality=95)
    print(f"Sol gerado: {out_path} ({rgb.shape[1]}x{rgb.shape[0]})")


def make_moon(tile, color_path, height_path):
    px, py, pz = sphere_grid(1024, 512)
    # mares lunares (escala global) + rugosidade fina do regolito
    maria = triplanar_fbm(tile, px, py, pz, octaves=(0.9, 1.9, 3.7))
    fine = triplanar_fbm(tile, -py, pz, px, octaves=(4.0, 8.5, 17.0, 31.0))

    specs = [
        (14, 0.18, 0.30, 1.10),   # grandes bacias de impacto
        (45, 0.08, 0.17, 0.85),   # crateras medias bem fundas
        (110, 0.035, 0.08, 0.62), # crateras nitidas espalhadas
        (180, 0.018, 0.035, 0.38) # micro-crateras
    ]
    craters, ejecta = generate_craters(px, py, pz, seed=42, specs=specs)

    # campo de elevacao total (normalizado em [0, 1])
    height = 0.22 * maria + 0.16 * fine + craters
    h_min, h_max = np.percentile(height, 1.0), np.percentile(height, 99.5)
    height_norm = np.clip((height - h_min) / (h_max - h_min + 1e-6), 0.0, 1.0)
    Image.fromarray((height_norm * 255.0 + 0.5).astype(np.uint8), "L").save(height_path)

    # cor (albedo lunar realista: mares mais escuros, terras altas e raios de ejeccao claros)
    maria_smooth = np.clip((maria - 0.38) / 0.32, 0.0, 1.0)
    base_albedo = 0.38 + 0.34 * maria_smooth + 0.10 * (fine - 0.5) + 0.18 * ejecta
    # leve escurecimento no fundo das bacias profundas
    base_albedo += 0.06 * np.clip(craters, -0.8, 0.4)
    base_albedo = np.clip(base_albedo, 0.20, 0.94)

    rgb = np.stack([
        base_albedo * 1.01,
        base_albedo * 1.00,
        base_albedo * 0.97,
    ], axis=-1)
    rgb = np.clip(rgb, 0.0, 1.0)
    Image.fromarray((rgb * 255.0 + 0.5).astype(np.uint8), "RGB").save(color_path, quality=95)
    print(f"Lua gerada: {color_path} e {height_path}")


def make_mercury(tile, color_path, height_path):
    px, py, pz = sphere_grid(1024, 512)
    terrain = triplanar_fbm(tile, pz, px, py, octaves=(1.2, 2.5, 5.2, 11.0))
    fine = triplanar_fbm(tile, px, -pz, py, octaves=(5.0, 10.0, 20.0))

    specs = [
        (16, 0.16, 0.28, 1.05),
        (55, 0.07, 0.15, 0.82),
        (130, 0.03, 0.07, 0.58),
    ]
    craters, ejecta = generate_craters(px, py, pz, seed=108, specs=specs)

    height = 0.25 * terrain + 0.15 * fine + craters
    h_min, h_max = np.percentile(height, 1.0), np.percentile(height, 99.5)
    height_norm = np.clip((height - h_min) / (h_max - h_min + 1e-6), 0.0, 1.0)
    Image.fromarray((height_norm * 255.0 + 0.5).astype(np.uint8), "L").save(height_path)

    albedo = 0.42 + 0.28 * (terrain - 0.3) + 0.10 * (fine - 0.5) + 0.15 * ejecta
    albedo = np.clip(albedo, 0.22, 0.90)
    # tom acinzentado-bronze tipico de Mercurio
    rgb = np.stack([
        albedo * 1.08,
        albedo * 0.96,
        albedo * 0.82,
    ], axis=-1)
    rgb = np.clip(rgb, 0.0, 1.0)
    Image.fromarray((rgb * 255.0 + 0.5).astype(np.uint8), "RGB").save(color_path, quality=95)
    print(f"Mercurio gerado: {color_path} e {height_path}")


def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    img_dir = os.path.join(root, "images")
    tile = load_tile_noise(os.path.join(img_dir, "noise.png"))

    make_sun(tile, os.path.join(img_dir, "sun.jpg"))
    make_moon(tile, os.path.join(img_dir, "moon.jpg"), os.path.join(img_dir, "moon_height.png"))
    make_mercury(tile, os.path.join(img_dir, "mercury.jpg"), os.path.join(img_dir, "mercury_height.png"))


if __name__ == "__main__":
    main()
