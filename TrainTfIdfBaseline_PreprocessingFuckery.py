import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import make_scorer, accuracy_score
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
#import language_tool_python
from concurrent.futures import ProcessPoolExecutor
import csv
import glob
from spellchecker import SpellChecker
import re

# Based on https://www.kaggle.com/competitions/llm-detect-ai-generated-text/discussion/456142
# but uses better regex that appears to handle ' the same way as the competition organizers.
def BuggyCorrectTypos(text):
    # Initialize spell checker
    spell = SpellChecker()
    # Tokenize the text into words
    words = re.findall(r"\b[\w|']+\b", text)

    # Find misspelled words
    misspelled = spell.unknown(words)

    # Correct the misspelled words
    corrected_text = text
    for word in misspelled:
        if (spell.correction(word)):
            corrected_text = corrected_text.replace(word, spell.correction(word))

    return corrected_text

# Based on https://www.kaggle.com/competitions/llm-detect-ai-generated-text/discussion/452172
# Removes anything in PERSUADE dataset that doesn't appear in "train.csv" from the competition organizers.
def RemoveBlacklistedCharacters(text):
    blacklisted_characters = ['\x94', ']', 'á', '¹', '`', 'å', '~', '}', 'Ö', '\\', '=', '\x97', '(', '©', '²', ')', '\x91', '>', '®', ';', '<', '£', '+', '#', '¶', '\xa0', '{', '^', '\x80', '[', '|', '\x93', '-', '\x85', 'Ó', '*', '/', '$', 'é', 'ó', '\x99']
    translation_table = str.maketrans('', '', ''.join(blacklisted_characters))
    cleaned_text = text.translate(translation_table)

    return cleaned_text

def RemoveTitle(text):
    # TODO: Regex like "[\w| ]{1,30}\n"
    # TODO: Sometimes moves "dear" onto previous line.
    pass

def ReplacePlacheholders(text):
    # TODO: PROPER_NAME *sometimes* replaced with actual name. Maybe only in "sincerely?"
    pass

def RemoveEverythingAfterLastPeriod(text):
    # TODO: They seem to remove trailing questions, exlamations, and signatures.
    pass

# def LoadData():
#     essays = []
#     labels = []

#     with open('data/PERSUADE/persuade_2.0_human_scores_demo_id_github.csv') as human_essays_file:
#         essays_reader = csv.DictReader(human_essays_file)
#         for row in essays_reader:
#             essays.append(row['full_text'])
#             labels.append(1)

#     for generated_essay_filepath in glob.glob('data/GeneratedEssays/*/*.json'):
#         # TODO: Pick up here.

#     return essays, labels

# if __name__ == '__main__':
#     pass

# original_essay = '''We currently live in a world dependent on machines; becoming more and more enveloped in this idea of reliance. However, now we have reached a point where that reliance is harming us and our planet, and it will continue to do so if we don't adjust our lifestyles. Seeing as that we've recently reached that point of realization, we are starting to limit ourselves to the things that do harm our planet, including limiting the usage of cars. Living in the twentieth century almost inevitably means relying on cars for transportation. Although it may seem impossible to convince so many to pursue this act of limiting the usage of cars, it is surely not impossible, as we've seen from the recent efforts made around the world. It will take time to adjust to the drastic change, but the benefits are worth the while. A few of the many advantages include, the conservation of our valuable space, improving the quality of our environment, and allowing for the opportunity of new ideas, developments, and change.

# The first of the advantages may not be the first you would think of, yet it is certainly an important one. By limiting our use of cars it would save and replenish our amount of space that we currently have available. It's not only the cars themselves that take up space, but it is also the space left for street parking in the big cities, the massive parking garages used at shopping malls, and the driveways and garages at the houses in large suburban neighborhoods. This method of limiting the use of cars as been successfully applied to a town in Germany, known as Vauban. In Vauban, '' [r]esidents of this upscale community are suburban pioneers, going where few soccer moms or commuting executives have ever gone before: they have given up their cars '' ( '' In German Suburb, Life Goes On Without Cars'' ). The town of Vauban, has given up their areas of street parking, driveways, and home garages to limit--almost entirely--their use of cars. With the removal of all traces of automobile usage, some may be concerned about how they are expected to reach their desired destinations, but in Vauban and many other places preparing the follow this plan, stores will be placed '' a walk away, on a main street, rather than in malls along some distant highway '' ('' In German Suburb, Life Goes On Without Cars'' ). Although some may be concerned with the potential limit on their car usage, they must be reminded that the space originally used for cars will certainly not go to waste, and will be used for bigger and better purposes that will not harm our planet.

# This second advantage is the one most associated with the limit put on our use of cars--the improved quality of the environment. The environment can be improved with the decrease of greenhouse gas emissions, which in turn improves the quality of the air that surrounds us. Our President, Barack Obama, has '' ambitious goals to curb the United States' greenhouse gas emissions '' ( '' The End of Car Culture '' ), but that can only be done with the cooperation of the citizens. Fortunately, that cooperation has been seen with the '' fortuitous assist from an incipient shift in American behavior '', in which '' recent studies suggest that Americans are buying fewer cars, driving less and getting fewer licenses as each year goes by '' ( '' The End of Car Culture '' ). It is understood by professionals, that if the pattern continues, '' it will have beneficial implications for carbon emissions and the environment, since transportation is the second largest source of America's emissions '' ( '' The End of Car Culture '' ). Although some may be reluctant to pursue the path of limiting their use of cars, they should be well informed that by doing so, they are improvong their overall state of living by not producing these harmful properties that are being released into the air that we breathe.

# This last advantage is most likely the most considerably accepted by the public, because of its simplicity on their part and the benefits they recieve from it. With the limit put on the use of cars, that allows for the opportunity of new ideas, developments, and change. The idea of limiting the use of cars has gone global, from Germany to Colombia to France, this idea is quickly becoming one widely accepted by the public and their officials. The idea of a car-free dat has sprouted from Colombia, in which the citizens of Colombia are encouraged tohike, bike, skate, or take the bus to work rather than using cars for transportation ( '' Car-free day is spinning into a big hit in Bogota '' ). The public has responded positively to these recent changes, saying, '' ' It's a good opportunity to take away stress and lower air pollution ' '', which was spoken by '' Carlos Arturo Plaza as he rode a two-seat bicycle with his wife '' ( '' Car-free day is spinning into a big hit in Bogota '' ). Not only have new opportunities for the public been introduced, but so have new developments. The new developmets include '' [p]arks and sports centers...uneven, pitted sidewalks have been replaced by broad, smooth sidewalks...and new restaurants and upscale shopping districts '' ( '' Car-free day is spinning into a big hit in Bogota '' ). The citizens of various cities can also expect to see plans in which '' ' pedestrian, bicycle, private cars, commercial and public transportation traffic are woven into a connected networl to save time, conserve resources, lower emissions, and improve safety ' '' ( '' The End of Car Culture '' ).

# There have been many advantages associated with a limit put on the use of cars, including the conservation of our valuable space, the lowering of the greenhouse gas emissions, and the opportunity for new ideas, developments, and change. All of these advantages can be fulfilled to their entire purposes, but it is a group effort as a planet. We can continue to live in the luxury of these advantages, as long as we do our part to limit our use of cars.'''

# original_essay = '''The electoral college is something that has been loved, hated, and debated on for some years now. We couold abolish it or keep it the way it is. Some say the electoral college is undemocratic and unfair to the voters and the candidates. But some say it is the best way to vote. I'ts seen both ways but is there anything we can do about it? Good or Bad.

# The Electoral College is a compromise between election of the President by a popular vote of qualified citizens. The group of electors for your state are selected by the cnadidate""s political party. So when you vote for a specific candidate you are voting for the electors he is supported by.

# So why do people want to aboplish the electoral college completely? Because there are some serious things wrong with it. ""perhaps most worrying is the prospect of a tie in the electoral vote"" (Source 2, Paragraph 4). How can there be a tie in the election of the president? Perhaps because sometimes the electors get to thinking in another midset and vote for the wrong candidate...Yes that is poosible. When people vote for there candidate the electors are the ones being selected and aren't always the way they were in their chosing so therefore there can be a serious tip in votes if the electors don't vote for the candidate of the party they were selected by. ""In 2000, taken shortly after Al Gore-thanks to the quirks of the electoral college-won the popular vote but lost the presidency, over 60 percent of voters would prefer a diect election"" (Source 2, Paragrph 1). The people don't want the electoral college anymore, Who's to say they ever did?

# But even know some don't want the electoral college there are some who prefer to keep it. ""each party selects a slate of electors trusted to vote for the party's nominee, and that trust is rarely betrayed"" (Source 3, Paragraph 2). So when the party selects their electors they are most likely to select the candidate of the party they were selected by. There is also regional appeal,if a president only has diserably mojority of vores in a certain region of the country he is ""unlikely to be a succesful president""( Source 3, Paragraph 5) . This is because if a president is only desired by a certain region and not the rest of the country, the wants of the rest of the country aren't met and he most likely will not tend to the needs of them, and ultimately doesn't have a chance of being selected as president. Finally, when no candidate for president has a clear majority of votes, the electoral college's votes gove a clear winner and a candidate is selected.

# Concluding, both sides have reasonable arguments to if the elctoral college will stay for good or be abolished completely. But it's up tp you to decide which side your own because this is a democracy.. Isn't it?'''

original_essay = '''Dear Mrs. Senator,

The Electoral College is unfair, outdated, and a poorly representative system for our nation. Previous elections and facts show that the Electoral College may have worked in the past, but does not work in accuratley representing the millions of voters in our country any longer.

In the 2000 presidential campaign, the unfairness of the Electoral College was blatantly obvious. ""Seventeen states didn't see the candidates at all, and voters in twenty five of the largest media markets didn't get to see a single campaig ad,"" (Plumer). The vote was left almost entirely in the hands of a few ""swing voters"" in Ohio, which is not an accurate representation of the opinions of the American population. During this campaign in 2000, Al Gore received more individual votes than George W. Bush nationwide, however, Bush received 271 electoral votes to Gore's 266,  so Bush was elected president (Plumer). It is obvious that the votes of the Electoral College do not accurately reflect the opinions of the people, and gives the citizens of our country poor representation in our government.

The arguments in favor of the Electoral College are weak at best. In the article ""In Defense of the Electoral College: Five reasons to keep our despised method of choosing the President"" by Richard A. Posner, the opening paragraph is practically an argument against the Electoral College. In the article, Posner states, ""The Electoral College is widely regarded as an anachronism, a non-democratic method of selecting a president... [t]he advocates of the position are correct in arguing that the Electoral College method is not democratic... it is the electors that choose the president, not the people."" In this opening statement for an article about how great the Electoral College is, Posner proves quite the opposite. The people of our country deserve proper representation; each of their individual votes should be important and their opinions on who leads this country should be heard.

As stated in this counterclaim, it is not the people who choose the president- it is the electors in the unjust Electoral College system.

According to a Gallup poll taken in 2000, over 60% of voters would prefer a direct election to the kind we have now (Plumer). It is clear the majority of the U.S.A. would prefer a different way of electing a president, whether that be a direct election, or another system that properly represents the nation's opinions. The list of possibilities of things going wrong in the Electortal College is large. Suppose there was a tie, which is entirely possible, since there is an even number, 538, of Electoral votes (Posner). If this happened, the election of the president would be put in the hands of the House of Representatives. In this system, each state casts only one vote, so the representative from Wyoming who is casting a vote for 500,000 voters would have as much say as the representative from California, whose single vote represents the opinion of 35 million voters . If this isn't an unethical process, I don't know what is.

Mrs. Senator, as a citizen of the United States, and Democrat in the mainly Republican state of Florida, I want my voice to be heard.

With the current Electoral Collge system, I am worried that it won't be. I shouldn't have to worry about ""faithless"" electors refusing to vote for my party's candidate, or the electors in my state not casting a vote that represents my opinions. In the free country of America, I want my vote to count and our government to make decisions based on what

I say, not some electors from my state. With this current Electoral College system, I cannot have my representation or freedom bestowed upon me by the Constitution, and that needs to change. Thank you for reading, and I hope you can make a difference.

Sincerely,

PROPER_NAME'''

print(BuggyCorrectTypos(original_essay))