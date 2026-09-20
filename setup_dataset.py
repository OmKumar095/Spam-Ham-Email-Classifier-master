import os
import shutil
import urllib.request
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'DataSets', 'data')
INDEX_DIR = os.path.join(BASE_DIR, 'DataSets', 'index')
TRAIN_CSV = os.path.join(INDEX_DIR, 'train.csv')
TRAIN_BAK = os.path.join(INDEX_DIR, 'train.csv.original')

DATASET_URL = 'https://raw.githubusercontent.com/bendgame/nlpBeginnerProjects/master/emails.csv'

def main():
    print("Setting up dataset...")
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(INDEX_DIR, exist_ok=True)

    if os.path.exists(TRAIN_CSV) and not os.path.exists(TRAIN_BAK):
        shutil.copyfile(TRAIN_CSV, TRAIN_BAK)

    print(f"Downloading email dataset from {DATASET_URL}...")
    df = pd.read_csv(DATASET_URL)
    print(f"Downloaded {len(df)} emails ({sum(df['spam'] == 1)} spam, {sum(df['spam'] == 0)} ham).")

    train_lines = []
    print(f"Writing email files to {DATA_DIR}...")

    for i, row in df.iterrows():
        file_num = i + 1
        label = "spam" if row['spam'] == 1 else "ham"
        filename = f"inmail.{file_num}"
        filepath = os.path.join(DATA_DIR, filename)

        raw_text = str(row['text'])
        # Separate Subject header from body so email.message_from_file parses both
        if raw_text.lower().startswith("subject:"):
            content = raw_text[8:].strip()
            # First line as subject, rest as body
            lines = content.split("  ", 1)
            subject = lines[0]
            body = lines[1] if len(lines) > 1 else lines[0]
            email_content = f"Subject: {subject}\nFrom: sender@example.com\nTo: recipient@example.com\n\n{body}"
        else:
            email_content = f"Subject: Notification\nFrom: sender@example.com\nTo: recipient@example.com\n\n{raw_text}"

        with open(filepath, "w", encoding="utf-8", errors="ignore") as f:
            f.write(email_content)

        train_lines.append(f"{label} ../data/{filename}\n")

        if (i + 1) % 1000 == 0 or (i + 1) == len(df):
            print(f"Generated {i + 1}/{len(df)} email files...")

    print(f"Writing updated index to {TRAIN_CSV}...")
    with open(TRAIN_CSV, "w", encoding="utf-8") as f:
        f.writelines(train_lines)

    print("Dataset setup completed successfully!")

if __name__ == '__main__':
    main()
