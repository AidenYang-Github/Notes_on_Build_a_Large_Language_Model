import urllib.request

url = ("https://raw.githubusercontent.com/rasbt/"
       "LLMs-from-scratch/main/ch02/01_main-chapter-code/"
       "the-verdict.txt")
file_path = "the-verdict.txt"
urllib.request.urlretrieve(url, file_path)

# 通过Python读取短篇小说The Verdict作为文本样本
with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()
print(f"Total number of character:{len(raw_text)}")
print(raw_text[:99])
"""
Total number of character:20479
I HAD always thought Jack Gisburn rather a cheap genius--though a good fellow enough--so it was no
"""

import re
text = "Hello, world. This, is a test."
result = re.split(r'(\s)', text)
print(result)