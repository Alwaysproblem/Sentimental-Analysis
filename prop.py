# import re 
from implementation import preprocess

stop_words = set({'ourselves', 'hers', 'between', 'yourself', 'again',
                  'there', 'about', 'once', 'during', 'out', 'very', 'having',
                  'with', 'they', 'own', 'an', 'be', 'some', 'for', 'do', 'its',
                  'yours', 'such', 'into', 'of', 'most', 'itself', 'other',
                  'off', 'is', 's', 'am', 'or', 'who', 'as', 'from', 'him',
                  'each', 'the', 'themselves', 'below', 'are', 'we',
                  'these', 'your', 'his', 'through', 'don', 'me', 'were',
                  'her', 'more', 'himself', 'this', 'down', 'should', 'our',
                  'their', 'while', 'above', 'both', 'up', 'to', 'ours', 'had',
                  'she', 'all', 'no', 'when', 'at', 'any', 'before', 'them',
                  'same', 'and', 'been', 'have', 'in', 'will', 'on', 'does',
                  'yourselves', 'then', 'that', 'because', 'what', 'over',
                  'why', 'so', 'can', 'did', 'not', 'now', 'under', 'he', 'you',
                  'herself', 'has', 'just', 'where', 'too', 'only', 'myself',
                  'which', 'those', 'i', 'after', 'few', 'whom', 't', 'being',
                  'if', 'theirs', 'my', 'against', 'a', 'by', 'doing', 'it',
                  'how', 'further', 'was', 'here', 'than',
                  'wouldn', 'shouldn', 'll', 'aren', 'isn'})

prostr = """It seems ever since 1982, about every two or three years we get a movie that claims to be "The Next Officer and a Gentleman." There has yet to be one movie that has lived up to this claim and this movie is no different.<br /><br />We get the usual ripped off scenes from OAAG ("I want you DOR," the instructor gives the Richard Gere character his overdose of drills in hopes he'll quit, the Gere character comes back for the girl, the Gere character realizes the instructor is great, etc.) and this movie is as predictable as the sun rising in the East and is horribly miscast on top. Costner plays his usual "wise teacher" character, the only character he can play, and you really get a sense of his limited acting abilities here. Kutcher is terrible in the Richard Gere character, just miscast with acting skills barely a notch above Keanu Reeves.<br /><br />The main problem with this OAAG wannabe is the two main characters are so amazingly one-dimensional, you never care for either in the least and when Kutcher's character finally turns around (just like Gere did in OAAG) you just go "so what? The movie leaves no plot point unturned and seems to never end as if to say "oh wait, we forgot to close out the girlfriend story, or the what happens after he graduates story, or the other six plot points in the movie..." What's more baffling is the great "reviews" I see here. The general public's opinions never cease to amaze me."""

# prostr = """It seems ever since 1982, about every different.<br /><br />We "wise teacher" you aren't fool."""


# def preprocess(review):
#     import re
#     page = r"<.*?>"

#     # pieces_nopage = re.sub(page, "", prostr)
#     pieces_nopara = re.compile(page).sub("", review)
#     # print(re.findall(page, review))
#     # print(pieces_nopara)

#     patten = r"\W+"
#     # pieces = re.split(patten, pieces_nopage)
#     pieces = re.compile(patten).split(pieces_nopara)

#     # print(pieces)

#     piece = [p.lower() for p in pieces if p != '' and p.lower() not in stop_words]
#     processed_review = " ".join(piece)

#     print(review, end = "\nafter:\n")
#     print(processed_review)

#     return processed_review

def main():
    processed_review = preprocess(prostr)
    print(processed_review)

if __name__ == '__main__':
    main()