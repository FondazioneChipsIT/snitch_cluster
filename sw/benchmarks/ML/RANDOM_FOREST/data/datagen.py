import random
from pathlib import Path

RF_N_FEATURES = 3
RF_N_CLASSES = 3
RF_N_TREES = 6

# Ogni tree avrà una dimensione compresa tra 280 e 300,
# preferibilmente divisibile per 8 e diversa dalle altre.
MIN_TREE_SIZE = 280
MAX_TREE_SIZE = 300

random.seed(42)


class DTNode:
    def __init__(self, feature, threshold, left, right, cls):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.cls = cls

    def emit(self):
        return f"  {{ {self.feature}, {self.threshold}, {self.left}, {self.right}, {self.cls},{{0,0,0,0,0,0}} }},"


def generate_tree_size(tree_idx, used_sizes):
    """
    Genera una dimensione tra 280 e 300.
    Preferisce valori divisibili per 8 e cerca di non ripetere.
    """

    preferred = [x for x in range(MIN_TREE_SIZE, MAX_TREE_SIZE + 1) if x % 8 == 0]
    fallback = list(range(MIN_TREE_SIZE, MAX_TREE_SIZE + 1))

    random.shuffle(preferred)
    random.shuffle(fallback)

    for size in preferred:
        if size not in used_sizes:
            return size

    for size in fallback:
        if size not in used_sizes:
            return size

    # Se finiscono i valori unici, accetta ripetizioni
    return random.choice(preferred if preferred else fallback)


def generate_tree_nodes(size):
    """
    Genera un tree sintetico valido con 'size' nodi.
    I nodi foglia hanno feature=-1.
    I nodi interni puntano a figli successivi.
    """

    nodes = []

    for i in range(size):
        # Ultimo ~40% come foglie
        if i >= int(size * 0.6):
            cls = random.randint(0, RF_N_CLASSES - 1)
            nodes.append(DTNode(-1, 0, -1, -1, cls))
        else:
            feature = random.randint(0, RF_N_FEATURES - 1)
            threshold = random.randint(0, 512)

            # Garantisce indici figli validi e crescenti
            left = min(i + 1, size - 1)
            right = min(i + random.randint(2, 8), size - 1)

            # Evita che un nodo interno punti a sé stesso
            if right <= left:
                right = min(left + 1, size - 1)

            nodes.append(DTNode(feature, threshold, left, right, -1))

    return nodes


def emit_header():
    used_sizes = set()
    tree_sizes = []

    for t in range(RF_N_TREES):
        size = generate_tree_size(t, used_sizes)
        used_sizes.add(size)
        tree_sizes.append(size)

    out = []
    out.append("// Auto-generated Random Forest model (integer Q8.8)")
    out.append("#pragma once")
    out.append("#include <stdint.h>")
    out.append("")
    out.append(f"#define RF_N_FEATURES {RF_N_FEATURES}")
    out.append(f"#define RF_N_CLASSES {RF_N_CLASSES}")
    out.append(f"#define RF_N_TREES {RF_N_TREES}")
    out.append("")

    out.append(
        "int16_t x_dram[RF_N_FEATURES] = { "
        "(int16_t)(0.90f*256), (int16_t)(0.10f*256), (int16_t)(1.00f*256) };"
    )
    out.append("")

    for t in range(RF_N_TREES):
        out.append(
            f"#define RF_TREE_{t}_SIZE "
            f"((uint16_t)(sizeof(RF_TREE_{t}_dram) / sizeof(RF_TREE_{t}_dram[0])))"
        )

    out.append("")

    for t, size in enumerate(tree_sizes):
        nodes = generate_tree_nodes(size)

        out.append(f"static const DTNode RF_TREE_{t}_dram[] = {{")
        for node in nodes:
            out.append(node.emit())
        out.append("};")
        out.append(f"// RF_TREE_{t}_SIZE = {size}")
        out.append("")

    return "\n".join(out)


if __name__ == "__main__":
    header = emit_header()
    output_path = Path(__file__).resolve().parent / "data.h"
    output_path.write_text(header)
    print(f"Creato file header in: {output_path}")

