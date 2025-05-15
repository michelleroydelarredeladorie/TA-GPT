#!/usr/bin/env python
import csv
from io import StringIO

from application import db, User, app  # adjust the import if needed

def main():
    with app.app_context():
        # Ensure the tables exist.
        db.create_all()

        print("Paste your Excel table data below.")
        print("Each row should have the name in the first column and the id in the second (tab-delimited).")
        print("When you are finished, just press Enter on an empty line.\n")

        # Read multi-line input until an empty line is entered.
        lines = []
        while True:
            line = input()
            if not line.strip():
                break
            lines.append(line)

        # Join the lines into a single string and use csv.reader to handle tab-separated values.
        data = "\n".join(lines)
        reader = csv.reader(StringIO(data), delimiter="\t")

        # Process each row in the pasted data.
        for row in reader:
            if len(row) < 2:
                print(f"Skipping incomplete row: {row}")
                continue

            name = row[0].strip()
            id_str = row[1].strip()

            try:
                # Convert the id from string to integer.
                user_id = int(id_str)
            except ValueError:
                print(f"Invalid id '{id_str}' in row {row}. Skipping this row.")
                continue

            # Create a new user and add it to the session.
            new_user = User(id=user_id, name=name)
            db.session.add(new_user)
            print(f"Added user: {name} with id: {user_id}")

        # Commit all changes to the database.
        db.session.commit()
        print("\nAll users have been successfully added to the database.")

if __name__ == "__main__":
    main()

