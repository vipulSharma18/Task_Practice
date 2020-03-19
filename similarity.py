import json
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
#import any other depency to run vectorizer

#i'm assuming you have already normalized the input files
file_open = open("issues_open.json", 'r')
issues_open = json.loads(file_open.read())

file_closed = open("issues_closed.json", 'r')
issues_closed = json.loads(file_closed.read())

file_combined = open("issues_combined.json", 'r')
issues_combined = json.loads(file_combined.read())

# extracting only the bodies of issues from the dictionaries, the all_bodies is ordered
# the list are ordered in the order of insertion, i checked it: https://stackoverflow.com/questions/13694034/is-a-python-list-guaranteed-to-have-its-elements-stay-in-the-order-they-are-inse

all_bodies = []
# to check duplicacy in the issues_combined only, remove this for loop from code
for issue, body in issues_open.items():
	all_bodies.append(body)
for issue, body in issues_combined.items():
	all_bodies.append(body)

# there is redundancy in this all_bodies list but to keep code logically lucid it is better to do this way, redundancy won't affect the similarity scores in any way
vectorizer = TfidfVectorizer(tokenizer=None)   #i'm assuming you have already normalized the input files

X = vectorizer.fit_transfrom(all_bodies).todense()  
# X now contains the transformed representation of each issue body, now we can directly access this transformed representation instead of
# repeatedly transforming each comment

#renaming reference for me, to prevent refactoring of code
open_dict = issues_open
closed_dict = issues_closed
combined_dict = issues_combined

pos_open = 0   # the index of the open issue currently in consideration
for opn_issue_num, open_body in open_dict.items():
	pos_combined = 0   # the index of the combined issue currently in consideration
	for cmb_issue_num, combined_body in combined_dict.items():
		if(cmb_issue_num == opn_issue_num):
      continue
    #2d array's row from 0 to len(open_dict) - 1, were of open issues' comments' transformation
    # to get actual index of combined issues's result of transformation in the X 2d array
    X_cmb_position = len(open_dict) + pos_combined 
    
    subset_matrix = [X[pos_open], X[X_cmb_position]]    #2d array of the result of vectorizer of only the issues in consideration
		pos_combined+=1
    
		#calculation of similarity
		sim = ((subset_matrix * subset_matrix.T).A)[0,1]  #actual computation on the result of vectorizer model
    
		if sim > 0.5:
			print(opn_issue_num, 'similar to', cmb_issue_num)
	pos_open+=1
