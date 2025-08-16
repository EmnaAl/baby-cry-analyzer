from pathlib import Path

cats = ['hungry','need_to_change','pain','tired']
total = 0
for cat in cats:
    count = len(list(Path(f"dataset/{cat}").glob("*.wav")))
    print(f"{cat}: {count} files")
    total += count

print(f"\nTotal files: {total}")
