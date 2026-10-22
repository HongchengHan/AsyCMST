"""Create reproducible stratified train/validation/test CSV files."""
import argparse
import csv
import random
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='CSV containing at least a label column')
    parser.add_argument('--output-dir', help='Defaults to the input CSV directory')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.is_file():
        raise FileNotFoundError(f'Label CSV not found: {input_path}')
    output_dir = Path(args.output_dir) if args.output_dir else input_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    with input_path.open(newline='', encoding='utf-8-sig') as stream:
        reader = csv.DictReader(stream)
        fieldnames = reader.fieldnames
        table = list(reader)
    if not fieldnames or 'label' not in fieldnames:
        raise ValueError(f"{input_path} must contain a 'label' column")

    required = {'bus_vid', 'ceus_vid', 'label'}
    if not required.issubset(fieldnames):
        raise ValueError(f'{input_path} must contain columns: {sorted(required)}')
    complete, incomplete = [], []
    for row in table:
        category = {'0': '0.ben', '1': '1.mal'}.get(row['label'])
        bus = input_path.parent / 'images' / str(category) / 'bus' / Path(row['bus_vid']).stem
        ceus = input_path.parent / 'images' / str(category) / 'ceus' / Path(row['ceus_vid']).stem
        if category and bus.is_dir() and ceus.is_dir() and any(bus.iterdir()) and any(ceus.iterdir()):
            complete.append(row)
        else:
            incomplete.append(row.get('id', '<unknown>'))
    table = complete
    if incomplete:
        print(f'Skipping {len(incomplete)} samples with missing/empty frame directories: '
              f'{", ".join(incomplete)}')

    # Paper protocol: train/validation/test = 5:2:3.
    generator = random.Random(args.seed)
    groups = {}
    for row in table:
        groups.setdefault(row['label'], []).append(row)
    train, valid, test = [], [], []
    for rows in groups.values():
        generator.shuffle(rows)
        train_end = round(len(rows) * 0.5)
        valid_end = train_end + round(len(rows) * 0.2)
        train.extend(rows[:train_end])
        valid.extend(rows[train_end:valid_end])
        test.extend(rows[valid_end:])
    for split in (train, valid, test):
        generator.shuffle(split)
    for name, split in [('train_1.csv', train), ('valid_1.csv', valid), ('test_1.csv', test)]:
        destination = output_dir / name
        with destination.open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(split)
        print(f'{destination}: {len(split)} samples')


if __name__ == '__main__':
    main()
