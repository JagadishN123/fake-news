# Data Exploration

# Check the dataset size
print(f'Dataset size: {data.shape}')

# View a few samples of the data
print(data.sample(5))

# Check for missing values
print(data.isnull().sum())

# Understand label distribution
print(data['label'].value_counts())

# Analyze text length
data['text_length'] = data['text'].apply(lambda x: len(x.split()))
print(data['text_length'].describe())

# Word frequency analysis (optional)
from collections import Counter
word_freq = Counter(" ".join(data['text']).split()).most_common(20)
print(word_freq)

# Data Visualization (optional)
import matplotlib.pyplot as plt
data['label'].value_counts().plot(kind='bar')
plt.show()
