import random

# Sample text data
text = "hello world this is a simple text generation example using machine learning"

# Split words
words = text.split()

# Build mapping
word_dict = {}

for i in range(len(words) - 1):
    if words[i] not in word_dict:
        word_dict[words[i]] = []
    word_dict[words[i]].append(words[i+1])

# Generate text
current_word = random.choice(words)
generated = [current_word]

for _ in range(10):
    if current_word in word_dict:
        next_word = random.choice(word_dict[current_word])
        generated.append(next_word)
        current_word = next_word
    else:
        break

print("Generated Text:")
print(" ".join(generated))