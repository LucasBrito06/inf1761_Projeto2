"""Monta a cruz 4x3 do cubemap de fundo (espaco) a partir de space.jpg.

O TexCube do framework fatia uma imagem em cruz (grade 4x3) nas 6 faces:
    linha 0:        .   topo  .    .
    linha 1:      esq  frente dir  tras
    linha 2:        .  baixo  .    .
Cada face recebe uma variante (espelhada/rotacionada) do mesmo campo de
estrelas, para que as faces nao se repitam de forma obvia. Como o fundo e
pontos brancos sobre preto, nao ha emenda visivel entre as faces.

Uso: python make_space_cross.py space.jpg space_cross.png
"""
import sys

import numpy as np
from PIL import Image


def main():
    src, dst = sys.argv[1], sys.argv[2]
    face = np.asarray(Image.open(src).convert("RGB"))
    n = face.shape[0]

    variants = [
        face,
        face[:, ::-1],
        face[::-1, :],
        face[::-1, ::-1],
        np.transpose(face, (1, 0, 2)),
        np.transpose(face, (1, 0, 2))[::-1, ::-1],
    ]

    background = face[0, 0]  # cor do ceu, para as celulas sem uso da cruz
    cross = np.tile(background, (3 * n, 4 * n, 1)).astype(np.uint8)

    # (linha, coluna) de cada face na cruz: topo, esq, frente, dir, tras, baixo
    cells = [(0, 1), (1, 0), (1, 1), (1, 2), (1, 3), (2, 1)]
    for (row, col), img in zip(cells, variants):
        cross[row * n:(row + 1) * n, col * n:(col + 1) * n] = img

    Image.fromarray(cross, "RGB").save(dst)
    print(f"{dst}: {cross.shape[1]}x{cross.shape[0]}")


if __name__ == "__main__":
    main()
