from transformers import BertTokenizer, BertModel, BertConfig

def split_tokenizer(text):
    return text.split()

def token_loc(text, tokenizer):
    """
    For a given tokenizer, obtain each token and its position in the original text
    """
    if isinstance(tokenizer, BertTokenizer):
        tokens = tokenizer.tokenize(text)
    else:
        tokens = tokenizer(text)
    token_word_loc = []
    i_pos = 0
    for token in tokens:
        tokenp = token
        if '▁' in token: # padding for word2piece
            if len(token) > 1:
                tokenp = token[1:]
            else:
                tokenp = ' '
        i_pos, f_pos = find_loc(text, tokenp, start_i=i_pos)
        temp = (token, (i_pos, f_pos))
        token_word_loc.append(temp)
        i_pos = f_pos
    return token_word_loc

def find_loc(text, subword, start_i=0):
    """
    In a text, gives index of subword from start_i
    in case subword is space, prohibit it matches with spaces 
    behind of sentences
    :return: 
    i_loc, f_loc
    """
    if subword == ' ':
        if text[start_i] == ' ':
            return start_i, start_i+1
        else:
            return start_i, start_i
    try:
        i_loc = start_i + text[start_i:].index(subword)
        f_loc = i_loc + len(subword)
    except:
        i_loc = start_i
        f_loc = i_loc
    
    return i_loc, f_loc
# test
#find_loc('I love youy love', 'love', start_i = 5)

import re
def get_NE(tag_text, text, ne_exp='<(.+?):([A-Z]{3})>'):
    """
    find Named Entity from given text
    :return: 
    [[NE, TAG, (i_pos, f_pos)]]
    """
    ne_word_loc = []
    reg_exp = re.compile(ne_exp)
    nes = reg_exp.finditer(tag_text)
    i_pos = 0
    for ne in nes:
        subword, tag = ne[1], ne[2]
        i_pos, f_pos = find_loc(text, subword, start_i=i_pos)
        temp = [subword, tag, (i_pos, f_pos)]
        ne_word_loc.append(temp)
    return ne_word_loc

def match(tag_tokens, model_tokens):
    """
    :in: 
    tokens1, tokens2 are in form of [(word, pos)], pos = (i_loc, f_loc)
    f_loc is inclusive
    :desc:
    if some forms are not adequate like empty subword, it can be removed by post-processing
    """
    matching_ids = []
    for tgt_token in tag_tokens:
        tgt_i_loc, tgt_f_loc = tgt_token[-1]
        i_id, f_id = 0, 0
        for i_token, token in enumerate(model_tokens):
            i_loc, f_loc = token[-1]
            # case1
            if tgt_i_loc >= i_loc and tgt_i_loc < f_loc:
                i_id = i_token
            # case2
            if tgt_f_loc > i_loc and tgt_i_loc <= f_loc:
                f_id = i_token
        matching_ids.append((i_id, f_id))
    return matching_ids

"""
def get_tag_text(text, tags):
    
    #both inputs are form of sequences
    
    tag_text = []
    for ii, token in enumerate(text):
        tag_token = "<%s:%s>"
        
"""

if __name__ == '__main__':
    # test
    # SKT KoBERT
    import gluonnlp as nlp
    from kobert.utils import get_tokenizer
    from kobert.pytorch_kobert import get_pytorch_kobert_model
    from transformers import BertTokenizer, BertModel, BertConfig
    
    # KoBERT tokenizer
    kobert, vocab = get_pytorch_kobert_model()
    tokenizer = nlp.data.BERTSPTokenizer(get_tokenizer(), vocab, lower=False)
    
    text = "오에 겐자부로는 일본 현대문학의 초석을 놓은 것으로 평가받는 작가 나쓰메 소세키(1867~1916)의 대표작 ‘마음’에 담긴 군국주의적 요소, 야스쿠니 신사 참배 행위까지 소설의 삽화로 동원하며 일본 사회의 ‘비정상성’을 문제 삼는다."
    tag_text = "<오에 겐자부로:PER>는 <일본:LOC> 현대문학의 초석을 놓은 것으로 평가받는 작가 <나쓰메 소세키:PER>(<1867~1916:DUR>)의 대표작 ‘<마음:POH>’에 담긴 군국주의적 요소, <야스쿠니 신사:ORG> 참배 행위까지 소설의 삽화로 동원하며 <일본:ORG> 사회의 ‘비정상성’을 문제 삼는다."
    tokens = tokenizer(text)
    print(tokens)
    
    ret = []
    for text1 in texts:
        ret.extend(tokenizer(text1))
    print(texts)
    print()
    print(ret)
    ret == tokens
    
    arr = '3417 6896  517 5402 7147 6398 6079 5760 3809 5051 6237 7095 4501'.split()
    for x in arr:
        print(tokenizer.vocab.idx_to_token[int(x)])
        
    ne_exp = '<(.+?):([A-Z]{3})>'
    ner_tokens = get_NE(tag_text, text, ne_exp)
    print(ner_ids)
    
    matching_ids = match(ner_tokens, wp_tokens)

    # preprocess (TAG --> B-TAG or I-TAG)
    tg_tags = [(i, x[1]) for i, x in enumerate(ner_tokens)]
    tags = [None]*len(wp_tokens)
    for ii, tag in tg_tags:
        i_id, f_id = matching_ids[ii]
        tags[i_id] = 'B-'+tag
        for jj in range(i_id+1, f_id+1):
            tags[jj] = 'I-'+tag

    # test
    for ii, tag in enumerate(wp_tokens):
        print(tags[ii], tag[0])