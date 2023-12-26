import stanza

class SyntaxAnalyzer:
    def __init__(self) -> None:
        self.pipeline = stanza.Pipeline(
            lang='ru', 
            use_gpu=True,
            package='syntagrus'
        ) 

    def __call__(self, text: str) -> stanza.Document:
        """Сегментация и синтаксический анализ текста

        Args:
            text (str): Входной текст

        Raises:
            AssertionError: при некорректных входных данных
            ValueError: при ошибке обработки текста

        Returns:
            stanza.Document: Список предложений с метками
        """
        assert type(text) is str
        assert len(text) != 0

        doc = self.pipeline(text)
        if doc is None:
            raise ValueError
        
        return doc

    @staticmethod
    def get_sentences(
        doc: stanza.Document,
        normalize: bool = False,
        upos: list[str] = []) -> list[str]:
        if normalize:
            attribute = 'lemma'
        else:
            attribute = 'text'

        sentences = []
        for sentence in doc.sentences:
            words = []
            for token in sentence.words:
                if attribute in dir(token):
                    cur_attr = attribute
                else:
                    cur_attr = 'text'

                if 'upos' in dir(token) and token.upos in upos or upos == []:
                    words.append(getattr(token, cur_attr))

            sentences.append(' '.join(words))
        
        return sentences
    
# NOUN - сущ, VERB или AUX - глагол, ADJ - прилагательное, 
# ADV - наречие, NUM - числительные, PROPN - местоимения,
# X - иностранные слова

# ppl = SyntaxAnalyzer()
# doc = ppl('Привет. Как дела? Я съел деда.')
# print(ppl.get_sentences(doc, normalize=True, upos=['NOUN', 'VERB', 'AUX']))