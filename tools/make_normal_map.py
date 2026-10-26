"""Gera um mapa de normais a partir de um campo de elevacao (luminancia da imagem).

Segue a formula da aula de rugosidade: com h o campo de elevacao e (s, t) os
eixos da textura,
    ps = [2, 0, h(i+1,j) - h(i-1,j)]
    pt = [0, 2, h(i,j+1) - h(i,j-1)]
    n  = ps x pt / |ps x pt|
e a normal e gravada como c = 0.5 n + 0.5 (dai o azul dominante: a normal
em repouso (0,0,1) vira (0.5, 0.5, 1)).

Convencao: a mesma do earth-normal.png do framework - verde para cima na
imagem, ou seja, t da formula cresce para CIMA.

Uso: python make_normal_map.py entrada.jpg saida.png [forca]
"""
import sys

import numpy as np
from PIL import Image


def height_to_normal_map(h, strength):
    # bordas: s repete (a esfera da a volta), t fica preso nos polos
    ds = np.roll(h, -1, axis=1) - np.roll(h, 1, axis=1)
    up = np.pad(h, ((1, 1), (0, 0)), mode="edge")
    dt = up[:-2, :] - up[2:, :]  # linha acima menos linha abaixo (t para cima)

    ps = np.stack([np.full_like(h, 2.0), np.zeros_like(h), ds * strength], axis=-1)
    pt = np.stack([np.zeros_like(h), np.full_like(h, 2.0), dt * strength], axis=-1)
    n = np.cross(ps, pt)
    n /= np.linalg.norm(n, axis=-1, keepdims=True)
    return n


def main():
    src, dst = sys.argv[1], sys.argv[2]
    strength = float(sys.argv[3]) if len(sys.argv) > 3 else 4.0
    img = Image.open(src).convert("L")
    h = np.asarray(img, dtype=np.float64) / 255.0
    n = height_to_normal_map(h, strength)
    rgb = np.clip((0.5 * n + 0.5) * 255.0 + 0.5, 0, 255).astype(np.uint8)
    Image.fromarray(rgb, "RGB").save(dst)
    print(f"{dst}: {rgb.shape[1]}x{rgb.shape[0]}, forca={strength}")


if __name__ == "__main__":
    main()
