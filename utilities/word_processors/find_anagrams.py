
import sys
import os
from collections import defaultdict

def find_txt_files(directory):
    """Recursively find all .txt files in the specified directory."""
    txt_files = []
    for subdir, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.txt'):
                txt_files.append(os.path.join(subdir, file))
    return txt_files

def read_words_from_file(file_path):
    """Read words from a file and return them as a list."""
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read().strip().split()

def find_anagrams(word_list):
    """Find anagrams in the given list of words and return them as a dictionary."""
    anagram_dict = defaultdict(list)
    for word in word_list:
        key = ''.join(sorted(word.lower()))
        anagram_dict[key].append(word)
    return {k: sorted(set(v)) for k, v in anagram_dict.items() if len(set(v)) > 1}

def main(directory):
    """Main function to process files, find anagrams, and print results."""
    word_files = find_txt_files(directory)
    all_words = []
    for file_path in word_files:
        all_words.extend(read_words_from_file(file_path))

    anagrams = find_anagrams(all_words)

    # Print anagrams and summary
    print("Anagrams found:")
    for key, group in anagrams.items():
        print(f"{key}: {group}")

    print("\nSummary:")
    print(f"Total words processed: {len(all_words)}")
    print(f"Total unique words: {len(set(all_words))}")
    print(f"Total number of anagram groups found: {len(anagrams)}")
    print(f"Max anagram set size: {max(len(words) for words in anagrams.values()) if anagrams else 0}")
    print(f"Total word files processed: {len(word_files)}")
if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python3 find_anagrams.py <path to directory>")
        sys.exit(1)
    main(sys.argv[1])

