"""
SignLanguageAI
Phase 2.3A

CSV Writer
"""

import csv
from pathlib import Path


class CSVWriter:

    def __init__(self, csv_path):

        self.csv_path = Path(csv_path)

        self.csv_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        if not self.csv_path.exists():

            self.create_csv()

    def create_csv(self):

        header = [
            "label",
            "hand"
        ]

        for i in range(21):

            header.extend([
                f"x{i}",
                f"y{i}",
                f"z{i}"
            ])

        with open(
            self.csv_path,
            "w",
            newline=""
        ) as file:

            writer = csv.writer(file)

            writer.writerow(header)

    def append(
        self,
        label,
        hand,
        landmarks
    ):

        row = [
            label,
            hand
        ]

        for landmark in landmarks:

            row.extend([
                landmark.x,
                landmark.y,
                landmark.z
            ])

        with open(
            self.csv_path,
            "a",
            newline=""
        ) as file:

            writer = csv.writer(file)

            writer.writerow(row)