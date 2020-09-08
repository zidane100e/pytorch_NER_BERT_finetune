class DecoderFromNamedEntitySequence():
    def __init__(self, tokenizer, index_to_ner):
        self.tokenizer = tokenizer
        self.index_to_ner = index_to_ner

    def change2tokens(self, in_ids):
        vocab = self.tokenizer.ids_to_tokens
        tokens = [ vocab[x] for x in in_ids ]
        tokens = [x[2:] if x[:2]=='##' else "▁"+x for x in tokens]
        tokens = [x[1:] if x in ['▁[CLS]', '▁[SEP]', '▁[PAD]'] else x for x in tokens ]
        return tokens
    
            
    def __call__(self, list_of_input_ids, list_of_pred_ids):
        input_tokens = self.change2tokens(list_of_input_ids)
        
        mask_ids = []
        for ii, token in enumerate(input_tokens):
            #print(token)
            if token not in ['[CLS]', '[SEP]', '[PAD]']:
                mask_ids.append(ii)
        #print('mask', mask_ids)
        
        input_tokens = [ input_tokens[ii] for ii in mask_ids ]
        list_of_input_ids = [ list_of_input_ids[ii] for ii in mask_ids ]
        list_of_pred_ids = [ list_of_pred_ids[ii] for ii in mask_ids ]
        
        pred_ner_tag = [self.index_to_ner[pred_id] for pred_id in list_of_pred_ids]

        #print("len: {}, input_tokens:{}".format(len(input_tokens), input_tokens))
        #print("len: {}, pred_ner_tag:{}".format(len(pred_ner_tag), pred_ner_tag))

        # ----------------------------- parsing list_of_ner_word ----------------------------- #
        list_of_ner_word = []
        entity_word, entity_tag, prev_entity_tag = "", "", ""
        for i, pred_ner_tag_str in enumerate(pred_ner_tag):
            if "B-" in pred_ner_tag_str:
                entity_tag = pred_ner_tag_str[-3:]

                if prev_entity_tag != entity_tag and prev_entity_tag != "":
                    list_of_ner_word.append({"word": entity_word.replace("▁", " "), "tag": prev_entity_tag, "prob": None})

                entity_word = input_tokens[i]
                prev_entity_tag = entity_tag
            elif "I-"+entity_tag in pred_ner_tag_str:
                entity_word += input_tokens[i]
            else:
                if entity_word != "" and entity_tag != "":
                    list_of_ner_word.append({"word":entity_word.replace("▁", " "), "tag":entity_tag, "prob":None})
                entity_word, entity_tag, prev_entity_tag = "", "", ""


        # ----------------------------- parsing decoding_ner_sentence ----------------------------- #
        decoding_ner_sentence = ""
        is_prev_entity = False
        prev_entity_tag = ""
        is_there_B_before_I = False

        for token_str, pred_ner_tag_str in zip(input_tokens, pred_ner_tag):
            token_str = token_str.replace('▁', ' ')  # '▁' 토큰을 띄어쓰기로 교체

            if 'B-' in pred_ner_tag_str:
                if is_prev_entity is True:
                    decoding_ner_sentence += ':' + prev_entity_tag+ '>'

                if token_str[0] == ' ':
                    token_str = list(token_str)
                    token_str[0] = ' <'
                    token_str = ''.join(token_str)
                    decoding_ner_sentence += token_str
                else:
                    decoding_ner_sentence += '<' + token_str
                is_prev_entity = True
                prev_entity_tag = pred_ner_tag_str[-3:] # 첫번째 예측을 기준으로 하겠음
                is_there_B_before_I = True

            elif 'I-' in pred_ner_tag_str:
                decoding_ner_sentence += token_str

                if is_there_B_before_I is True: # I가 나오기전에 B가 있어야하도록 체크
                    is_prev_entity = True
            else:
                if is_prev_entity is True:
                    decoding_ner_sentence += ':' + prev_entity_tag+ '>' + token_str
                    is_prev_entity = False
                    is_there_B_before_I = False
                else:
                    decoding_ner_sentence += token_str

        return list_of_ner_word, decoding_ner_sentence