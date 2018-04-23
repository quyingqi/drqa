#--*- coding:utf-8 -*--
from __future__ import division

import corenlp
import re
import gzip
import json

def strQ2B(ustring):
    """全角转半角"""
    rstring = ""
    for uchar in ustring:
        inside_code=ord(uchar)
        if inside_code == 12288:                              #全角空格直接转换
            inside_code = 32
        elif (inside_code >= 65281 and inside_code <= 65374): #全角字符（除空格）根据关系转化
            inside_code -= 65248

        # 2.7 unichr  
        rstring += chr(inside_code)
    return rstring

def wordFeature(client, text):
    ann = client.annotate(text)
    sentences = ann.sentence
    tokens = []
    pos = []
    ners = []
    beginChars = []
    endChars = []
    for sentence in sentences:
        for token in sentence.token:
            tokens.append(token.word)
            pos.append(token.pos)
            ners.append(token.ner)
            beginChars.append(token.beginChar)
            endChars.append(token.endChar)
    return tokens, pos, ners, beginChars, endChars


def removePunctuation(text):
    ## 2.7  should add .decode('utf8')
    return re.sub("[\s+\.\!\/_,$%:：^*(+\"\']+|[+——！，。？?、~@#￥%……&*（）]+", " ",text)

def removeHtmlLabel(text):
    dr = re.compile(r'<[^>]+>', re.S)
    return dr.sub('',text)

def readFile(filename):
    if '.gz' in filename:
        with gzip.open(filename, 'r') as pf:
            for line in pf:
                yield line
    elif '.json' in filename:
        with open(filename, 'r') as pf:
            for line in pf:
                yield line
def findsubstr(strs, target):
    """

    :param strs:  the strs to find str(target)
    :param target: the str to be found
    :return: the offset list
    """
    start = 0
    result = []
    while True:
        index = strs.find(target, start)
        if index == -1:
            break
        result.append(index)
        start = index + 1

    return result
def cal_endindex(target, starts):
    ends = []
    n = len(target)
    for start in starts:
        ends.append(start + n-1)
    return ends


def pointLabel(golden_answer, evidence_str, evidence_tokens, evidence_charstart):
    """
      :param golden_answer: List of answers, maybe the answer is not unique
      :param evidence_str:  evidence before  segmentation
      :param evidence_tokens:  evidence tokens after segmentation
      :param evidence_charstart: seg offset
      :return: starts [], ends[]
    """
    ## 如:  陶渊明   陶渊明 诗人
    ## 如： 双十协议  双十  协议
    ## 如：  桂花   咏桂花

    ## [['no_answer']]
    if len(golden_answer) == 1 and (golden_answer[0]) == 'no_answer':
        return [-1], [-1]
    else:
        starts = []
        ends = []
        for answer in golden_answer:
#    	    print(answer)
            start_indexs = findsubstr(evidence_str, answer)
            end_indexs = cal_endindex(answer, start_indexs)
            for idx, index_start in enumerate(start_indexs):
                index_end = end_indexs[idx]
                for i in range(len(evidence_tokens)):
		    
                    if evidence_charstart[i] <= index_start and index_start < len(evidence_tokens[i]) + evidence_charstart[i]:
                        starts.append(i)
                    if index_end < evidence_charstart[i] + len(evidence_tokens[i]) and evidence_charstart[i] <= index_end:
                        ends.append(i)
        return starts, ends


def qecomm_features(question, evidence_tokens):
    featurelist = [0 for _ in range(len(evidence_tokens))]
    for idx in range(len(evidence_tokens)):
        token = evidence_tokens[idx]
        if token in question:
            featurelist[idx] = 1
    return featurelist

def formate_baidudata(filepath, outputpath):
    result = open(outputpath, 'w')
    con = readFile(filepath)
    num = 0
    count = 0
    matched = 0
    with corenlp.CoreNLPClient(annotators='tokenize ssplit ner'.split()) as client:
        for line in con:
            num += 1
            line = line.strip()
            linedict = {}
            json_str = json.loads(line)
            q_key = json_str['q_key']
            question =''.join(json_str['question_tokens'])
            question = strQ2B(question)
            question_tokens, question_pos, question_ners, question_starts, question_ends = wordFeature(client, question)
            evidences = json_str['evidences']
            linedict['q_key'] = q_key
            #linedict['question'] = question
            linedict['question_tokens'] = question_tokens
            linedict['question_pos'] = question_pos
            linedict['question_ners'] = question_ners
            evidencelist = []
            frecounter = {}
            for evidence in evidences:
                evidencedict = {}
                e_key = evidence['e_key']
                evidencedict['e_key'] = e_key
                evidence_str = ''.join(evidence['evidence_tokens'])
                evidence_str = strQ2B(evidence_str)
                evidence_str = removeHtmlLabel(evidence_str)
                try:
                    evidence_tokens, evidence_pos, evidence_ner, evidence_charstart, evidence_charend = wordFeature(client, evidence_str)
                except:
                    continue

                for i in range(len(evidence_tokens)):
                    if not evidence_pos[i] == 'PU':
                        frecounter.setdefault(evidence_tokens[i], 0)
                        frecounter[evidence_tokens[i]] = frecounter[evidence_tokens[i]] + 1

                golden_answer = evidence['golden_answers']
                golden_answer_str = []
                for answer in golden_answer:
                    ans = ''.join(answer)
                    ans = strQ2B(ans)
                    ans = removeHtmlLabel(ans)
                    golden_answer_str.append(ans)

                starts, ends = pointLabel(golden_answer_str, evidence_str, evidence_tokens,evidence_charstart)
                qefeature = qecomm_features(question, evidence_tokens)
                #evidencedict['evidence_text'] = evidence_str
                evidencedict['evidence_tokens'] = evidence_tokens
                evidencedict['evidence_pos'] = evidence_pos
                evidencedict['evidence_ners'] = evidence_ner
                evidencedict['answer_starts'] = starts
                evidencedict['answer_ends'] = ends
                evidencedict['qecomm'] = qefeature
                evidencelist.append(evidencedict)

            for evi_dict in evidencelist:
                evidence_tokens = evi_dict['evidence_tokens']
                evidence_pos = evi_dict['evidence_pos']
                fre_tokens = []
                for i in range(len(evidence_tokens)):
                    token = evidence_tokens[i]
                    pos = evidence_pos[i]
                    if pos == 'PU':
                        fre_tokens.append(0)
                    else:
                        fre_tokens.append(frecounter[token])
                evi_dict['fre_tokens'] = fre_tokens
            linedict['evidences'] = evidencelist
            outputline = json.dumps(linedict)
            result.write(outputline + '\n')
            #if num % 1000 == 0:
            print(str(num) + " questions have been done!")


def formatsougoudata(filepath, istrain, outputpath):
    result = open(outputpath, 'w')
    con = readFile(filepath)
    with corenlp.CoreNLPClient(annotators='tokenize ssplit ner'.split()) as client:
        for line in con:
            linedict = {}
            line = line.strip()
            json_str = json.loads(line)
            query_id = json_str['query_id']
            query = json_str['query']
            if istrain:
                answer = json_str['answer']
            question_tokens, question_pos, question_ners, question_starts, question_ends = wordFeature(client, query)
            linedict['q_key'] = query_id
            linedict['question_tokens'] = question_tokens
            linedict['question_pos'] = question_pos
            linedict['question_ners'] = question_ners
            passages = json_str['passages']
            evidencelist = []
            frecounter = {}
            for passage in passages:
                evidencedict = {}
                passage_id = str(query_id) + '_' + str(passage['passage_id'])
                passage_text = passage['passage_text']
                passage_text = strQ2B(passage_text)
                passage_text = removeHtmlLabel(passage_text)
                evidencedict['e_key'] = passage_id
                try:
                    evidence_tokens, evidence_pos, evidence_ner, evidence_starts, evidence_ends = wordFeature(client, passage_text)
                except:
                    continue
                evidencedict['evidence_tokens'] = evidence_tokens
                evidencedict['evidence_pos'] = evidence_pos
                evidencedict['evidence_ners'] = evidence_ner
                for i in range(len(evidence_tokens)):
                    if not evidence_pos[i] == 'PU':
                        frecounter.setdefault(evidence_tokens[i], 0)
                        frecounter[evidence_tokens[i]] = frecounter[evidence_tokens[i]] + 1
                if istrain:
                    if not answer in passage_text:
                        starts, ends = pointLabel(['no_answer'], passage_text, evidence_tokens,evidence_starts)
                    else:
                        starts, ends = pointLabel([answer], passage_text, evidence_tokens, evidence_starts)

                    evidencedict['answer_starts'] = starts
                    evidencedict['answer_ends'] = ends

                qefeature = qecomm_features(query, evidence_tokens)
                evidencedict['qecomm'] = qefeature
                evidencelist.append(evidencedict)
            for evi_dict in evidencelist:
                evidence_tokens = evi_dict['evidence_tokens']
                evidence_pos = evi_dict['evidence_pos']
                fre_tokens = []
                for i in range(len(evidence_tokens)):
                    token = evidence_tokens[i]
                    pos = evidence_pos[i]
                    if pos == 'PU':
                        fre_tokens.append(0)
                    else:
                        fre_tokens.append(frecounter[token])
                evi_dict['fre_tokens'] = fre_tokens
                # print (evi_dict.keys())
            linedict['evidences'] = evidencelist
            outputline = json.dumps(linedict)
            result.write(outputline  + '\n')
            # print(linedict)


if __name__ == '__main__':
    #formatsougoudata('/media/data/fym/train_factoid_1.json',True, './sougou_train.json')
    #formate_baidudata('/media/data/fym/WebQA.v1.0/data/training.json.gz','baidu.json')
    #exit(0)

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--database', dest='database', type=str, help='database:baidu, sougou')
    parser.add_argument('--type',dest='type', type=str, help='data type: train,validation,test')
    parser.add_argument('--input_file',dest='input_path',type=str, help='filepath for the input data')
    parser.add_argument('--output_file', dest='output_path', type=str, help='filepath for the output data')

    args = parser.parse_args()
    input_path = args.input_path
    output_path = args.output_path
    type = args.type
    if args.database == 'baidu':
        formate_baidudata(input_path, output_path)

    elif args.database == 'sougou':
        if type == 'train' or type == 'valid':
            formatsougoudata(input_path, True, output_path)
        else:
            formatsougoudata(input_path, False, output_path)

    else:
        print("No such database!")



    # filename = '/Users/fuyanmei/Downloads/WebQA.v1.0/data/training.json.gz'
    # filename = '/Users/fuyanmei/Downloads/train_factoid_1.json'
    # outputpath = './sougou_train.json'
    # formatsougoudata(filename, True,outputpath)
    # #formate_baidudata(filename,'./baidudata.json')
