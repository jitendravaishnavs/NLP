import nltk

from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords


paragraph = """I have three visions for India.
In 3000 years of our history, people from all over the world have come and invaded us, captured our lands, conquered our minds.
From Alexander onwards, the Greeks, the Turks, the Moguls, the Portuguese, the British, the French, the Dutch — all of them came and looted us, took over what was ours.
Yet we have not done this to any other nation.
We have not conquered anyone.
We have not grabbed their land, their culture, their history and tried to enforce our way of life on them.
Why? Because we respect the freedom of others.

First Vision – Freedom
I have a vision. India must first stand up to the world.
Because I believe that unless India stands up to the world, no one will respect us.
Only strength respects strength.
We must be strong not only as a military power but also as an economic power.
Both must go hand in hand.

Second Vision – Development
My second vision for India is Development.
For fifty years we have been a developing nation.
It is time we see ourselves as a developed nation.
We are among the top five nations in the world in terms of GDP.
We have 10 percent growth rate in most areas.
Our poverty levels are falling.
Our achievements are being globally recognized today.
Yet we lack self-confidence that we are a developed nation.
We are still called a developing nation.
It is time we see ourselves as a developed nation.

Third Vision – India Must Stand Up
My third vision for India is that India must stand up to the world.
India must not be afraid to take bold decisions.
Because I believe unless India stands up to the world, no one will respect us.
We must become so strong that no one dares to look down upon us.
Only strength respects strength.
We must stand up as a self-reliant, self-assured nation.

India's Biggest Enemy – Our Own Lack of Confidence
Do we not realize that self-respect comes from self-reliance?
I believe that there are three members in society that destroy a nation:

People who do not respect their own culture.

People who do not respect their own heritage.

People who do not have confidence in themselves.

Be Courageous
Why are we so negative?
Why are we so embarrassed to recognize our own strengths, our achievements?
We are the first in milk production.
We are number one in remote sensing satellites.
We are the second largest producer of wheat.
We are the second largest producer of rice.
We have achieved self-sufficiency in many areas, yet we still lack confidence.
Instead of being proud of our achievements, we indulge in self-criticism."""

sentences = nltk.sent_tokenize(paragraph)
lemmatizer = WordNetLemmatizer()

# Tokenize and lemmatize each sentence
# Remove stopwords and non-alphabetic tokens

#Lemmetization

for i in range(len(sentences)):
    words = nltk.word_tokenize(sentences[i])
    words = [lemmatizer.lemmatize(word) for word in words if word not in set(stopwords.words('english')) ]
    sentences[i] = ' '.join(words)
    print(f"{i+1} {sentences[i]}")

# Print the lemmatized sentences