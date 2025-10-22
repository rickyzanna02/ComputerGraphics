#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import argparse
from pathlib import Path

def main(input_path, output_path, start_frame, end_frame):
    # Carica il file JSON originale
    with open(input_path, "r") as f:
        data = json.load(f)

    # Filtra solo i frame nell’intervallo richiesto
    filtered = {}
    for i in range(start_frame, end_frame + 1):
        key = f"frame_{i}"
        if key in data:
            new_index = i - start_frame + 1
            filtered[f"frame_{new_index}"] = data[key]

    print(f"Frame totali selezionati: {len(filtered)}")

    # Salva il nuovo JSON
    with open(output_path, "w") as f:
        json.dump(filtered, f, indent=2)

    print(f"✅ File salvato in: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filtra e rinomina frame da un file JSON.")
    parser.add_argument("input", type=Path, help="Percorso al file JSON di input")
    parser.add_argument("output", type=Path, help="Percorso al file JSON di output")
    parser.add_argument("--start", type=int, default=980, help="Frame iniziale (incluso)")
    parser.add_argument("--end", type=int, default=1370, help="Frame finale (incluso)")
    args = parser.parse_args()

    main(args.input, args.output, args.start, args.end)

# prende i json  di 12000 frame e tiene solo da 980 a 1370 (mocap7), rinominandoli: 980 diventa 1, 981 diventa 2 , ...
# eseguire sia per position che per rotation

#uso: python filter_frame.py .\position_data.json .\position_data_filtered.json --start 980 --end 1370
#     python filter_frame.py .\rotation_data.json .\rotation_data_filtered.json --start 980 --end 1370
