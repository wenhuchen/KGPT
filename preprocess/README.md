I list the steps to obtain the KB-Text mapping from Wikipedia.
 
- Wikipedia Text Dump
I downloaded the Wikipedia text from a preprocessed version of HotpotQA in https://hotpotqa.github.io/wiki-readme.html. But it only contains the English version, I’m not entirely sure how to get the dump for other languages.

- Wiki-Title to WikiData-Q-ID
For wiki-title -> wikidata-id, please use /data/wenhu/entity2text/preprocess/wiki-to-kbid.json. It's the official dump from WikiData site, however, since the Wikipedia title has many redirect links, it's like some of them are not covered in this set, then you need to use the Wikidata API provided in https://github.com/UKPLab/coling2018-graph-neural-networks-question-answering, search for "wikidata-access". 

- Construct the WikiData Sqlite DB
  1. Download latest-all.json.gz from https://dumps.wikimedia.org/wikidatawiki
  2. run ```python create_summarized.py``` to extract the useful information from it.
  3. run ```python create_db.py``` to create the database to store the WikiData information
 
- Requesting knowledge graph of a certain WikiData-Q-ID
You can request it from DB file, it's a SQLite database. The API for getting a knowledge graph is described in Database.py. You can see examples from /data/wenhu/entity2text/preprocess/step2.py
Other useful documentation (Optional)
  1. How to access WikiData:
  https://github.com/UKPLab/coling2018-graph-neural-networks-question-answering/blob/master/WikidataHowTo.md
  2. How to get WikiData Dump:
  https://www.mediawiki.org/wiki/Wikidata_Toolkit/Client
