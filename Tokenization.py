import nltk

# nltk.download()

paragraph = """In 3000 years of our history, people from all over the world have come and invaded us, captured our lands, conquered our minds. From Alexander onwards, the Greeks, the Turks, the Moguls, the Portuguese, the British, the French, the Dutch, all of them came and looted us, took over what was ours. Yet we have not conquered anyone. We have not grabbed their land, their culture, their history and tried to enforce our way of life on them.
Why? Because we respect the freedom of others.
That is why my first vision is that of freedom.
I believe that India got its first vision of this freedom in 1857, when we started the War of Independence. It is this freedom that we must protect and nurture and build on."""

sentences = nltk.sent_tokenize(paragraph)
words = nltk.word_tokenize(paragraph)
