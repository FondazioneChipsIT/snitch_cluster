import argparse
import os
import re
import sys

SINGLE_CYCLE_FP_OPS = {
    "fadd.", "fsub.", "fmul.", "fdiv.", "fsqrt.",
    "fmin.", "fmax.", "fabs.",
    "fcvt.", "feq.", "flt.", "fle.", "fgt.", "fge.",
    "fsgnj.", "fsgnjn.", "fsgnjx.",
    "fmsub.", "fnmadd.", "fnmsub.", "fmac.", "fmadd."
}

DOUBLE_CYCLE_FP_OPS = {}

TRACE_LINE_PATTERN = re.compile(
    r"^\s*(?:(\d+)\s+(\d+)\s+)?([A-Za-z])\s+(0x[0-9a-fA-F]+)\s+([a-zA-Z0-9_.]+)"
)

MCYCLE_PATTERN = re.compile(r"#;\s*mcycle\s*=\s*(0x[0-9a-fA-F]+|\d+)")

def hex_int(value):
    try:
        return int(value, 16)
    except ValueError:
        raise argparse.ArgumentTypeError(f"'{value}' is not a valid hexadecimal value.")

def parse_args():
    parser = argparse.ArgumentParser(
        description="Parses QuestaSim Trace files (core 0-7) to count FLOPS and cycles via mcycle CSR."
    )
    parser.add_argument(
        "--folder",
        type=str,
        required=True,
        help="Path to the folder containing trace_hart_0000x.txt files"
    )
    parser.add_argument(
        "--start",
        type=hex_int,
        required=True,
        help="Starting hexadecimal address, mcycle instruction (e.g., 0x80000310)"
    )
    parser.add_argument(
        "--end",
        type=hex_int,
        required=True,
        help="Ending hexadecimal address, mcycle instruction (e.g., 0x800004a0)"
    )
    return parser.parse_args()

def get_instruction_flops(mnemonic):
    for op in DOUBLE_CYCLE_FP_OPS:
        if mnemonic.startswith(op):
            return 2
    for op in SINGLE_CYCLE_FP_OPS:
        if mnemonic.startswith(op):
            return 1
    return 0

def parse_trace(file_path, start_pc, end_pc):
    total_flops = 0
    tstart = None
    tend   = None
    count  = False

    with open(file_path, 'r') as f:
        for line in f:
            match = TRACE_LINE_PATTERN.match(line)
            if not match:
                continue

            current_pc = int(match.group(4), 16)
            mnemonic   = match.group(5)

            if mnemonic == "csrr":
                mcycle_match = MCYCLE_PATTERN.search(line)
                if mcycle_match:
                    raw = mcycle_match.group(1)
                    mcycle_val = int(raw, 16) if raw.startswith("0x") else int(raw)
                    if current_pc == start_pc and tstart is None:  # ← non dipende da count
                        tstart = mcycle_val
                    if current_pc == end_pc:                       # ← idem
                        tend = mcycle_val

            if current_pc == start_pc:
                count = True
            if current_pc == end_pc:
                count = False

            if not count:
                continue

            total_flops += get_instruction_flops(mnemonic)

    total_cycles = (tend - tstart) if (tstart is not None and tend is not None) else 0
    return total_flops, total_cycles

def main():
    args = parse_args()

    folder = os.path.abspath(os.path.expanduser(args.folder))
    if not os.path.isdir(folder):
        print(f"Error: Folder '{folder}' not found or is not a directory.")
        sys.exit(1)

    print(f"Folder: {folder}")
    print(f"Range:  [{hex(args.start)} - {hex(args.end)}]")
    print("=" * 45)

    total_flops_all = 0
    total_cycles_all = 0
    missing = []

    for i in range(8):
        filename  = f"trace_hart_0000{i}.txt"
        file_path = os.path.join(folder, filename)

        if not os.path.exists(file_path):
            missing.append(filename)
            print(f"  Core {i}: WARNING - file not found, skipping.")
            continue

        flops, cycles = parse_trace(file_path, args.start, args.end)
        total_flops_all  += flops
        total_cycles_all += cycles

        print(f"  Core {i}  |  FP ops: {flops:>10}  |  Cycles: {cycles:>10}")

    print("=" * 45)
    print(f"  TOTAL   |  FP ops: {total_flops_all:>10}  |  Cycles: {total_cycles_all:>10}")
    print("=" * 45)

    if missing:
        print(f"\nMissing files: {', '.join(missing)}")

if __name__ == "__main__":
    main()