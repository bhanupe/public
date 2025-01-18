public
======

public libraries
### Commands Used
1. Clone the repository:
   `git clone https://github.com/bhanupe/public.git`
2. Change directory:
   `cd public`
3. Change remote origin:
   `git remote set-url origin <your-repo-url>`
4. Create a branch:
   `git checkout -b yt-upw-1`
5. Create folders:
   `mkdir utilities`
   `mkdir utilities/word_processors`
6. Find and move the file:
   `find . -name "find_anagrams.py"`
   `mv <path-to-find_anagrams.py> utilities/word_processors/`
7. Stage and commit the file:
   `git add utilities/word_processors/find_anagrams.py`
   `git commit -m "YT-UPW-1: Move find_anagrams.py to word_processors"`


### How to Use `find_anagrams.py`

1. Input File Organization:
   - Place input word files (e.g.,`.txt` files) in a specificfolder for processing with recursively spread in sub folders 
   - Each file should contain a list of words, one word per line.

2. Running the Script:
   - Navigate to the directory containing the script:
     cd utilities/word_processors
   - Run the script with the input file as an argument:
     python3 find_anagrams.py Words.txt
     
3. Examples of Input Files:
   - Download word lists from websites like:
     - https://www.levidromelist.com/levidrome-list/dictionary
     - https://www.keithv.com/software/wlist/
   - Ensure the files are plain text (`.txt`) and formatted as one word per line.

4. Limitations:
   - Supported: `.txt` files with UTF-8 encoding.
   - Not Supported: Binary files, PDFs, or other non-text formats. Avoid using files with unsupported characters or formats.