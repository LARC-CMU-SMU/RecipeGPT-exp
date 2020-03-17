import spacy
class spacy_extension:
    def __init__(self):
        self.spacy = spacy.load('en_core_web_lg')
        
    def ingr(self, lst):
        """ use spacy to process the ingredients

        Args:
          lst: A list of ingredient names

        Returns:
          root_ingr: A list of root nouns in the lemmatized form
          hl: A list that describes whether the input words is a root noun and its highlighting status
          
        """
        hl = [[{'text':x, 'highlight': None} for x in i.split(' ')] for i in lst]
        root_ingr = []
        for i, ingr in enumerate(lst):
            
            # if it is a uni-gram, keep the whole term
            if ' ' not in ingr:
                hl[i][0]['highlight'] = 'wrong'
                doc = self.spacy(ingr)
                root_ingr.append(doc[0].lemma_)
                
            # if it is 2+ gram, then use spacy
            else:
                phrase = 'Mix the %s and water.'%ingr
                doc = self.spacy(phrase)
                for chunk in doc.noun_chunks:
                    if chunk.text != 'water':
                        for j, word in enumerate(hl[i]):
                            if word['text'] == doc[chunk.end - 1].text:
                                hl[i][j]['highlight'] = 'wrong' 
                                root_ingr.append(doc[chunk.end - 1].lemma_)
        return root_ingr, hl
    
    def instr(self, directions):
        """ use spacy to process the directions
        
        Args:
          lst: A list of ingredient names

        Returns:
          instr: The raw output of spacy
          hl: A list that describes whether the input words is a root noun and its highlighting status
        """
        instr = self.spacy(directions)
        hl = [{'text': token.text, 'highlight': None} for token in instr]
        return instr, hl