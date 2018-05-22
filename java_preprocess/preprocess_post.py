# --*- coding:utf-8 -*--
import json

def get_evidence_tmp(q_len, evidence_tokens):
    mid = int(q_len/2)
    for index in range(len(evidence_tokens)):
        evidence_tmp = []
        for i in range(index - mid, index):
            if i < 0:
                continue
                # evidence_tmp.append("X")
                # break
            evidence_tmp.append(evidence_tokens[i])
        for i in range(index, index + q_len - mid):
            if i >= len(evidence_tokens):
                # evidence_tmp.append("X")
                break
            evidence_tmp.append(evidence_tokens[i])
        yield evidence_tmp

def distance_features(question_token, evidence_tokens):
    l = len(question_token)
    jaccard_feature = []
    edit_feature = []

    for evidence_tmp in get_evidence_tmp(l, evidence_tokens):
        jaccard = cal_jaccard_similar_rate(evidence_tmp, question_token)
        jaccard_feature.append(jaccard)

        edit = round((levenshtein(evidence_tmp, question_token)) / l, 2)
        edit_feature.append(edit)
    return jaccard_feature, edit_feature


def levenshtein(first, second):
    if len(first) > len(second):
        first, second = second, first
    if len(first) == 0:
        return len(second)
    if len(second) == 0:
        return len(first)
    first_length = len(first) + 1
    second_length = len(second) + 1
    distance_matrix = [list(range(second_length)) for x in range(first_length)]

    for i in range(1, first_length):
        for j in range(1, second_length):
            deletion = distance_matrix[i - 1][j] + 1
            insertion = distance_matrix[i][j - 1] + 1
            substitution = distance_matrix[i - 1][j - 1]
            if first[i - 1] != second[j - 1]:
                substitution += 1
            distance_matrix[i][j] = min(min(insertion, deletion), substitution)

    return distance_matrix[first_length - 1][second_length - 1]


def test_editdistance():
    question_tokens = ["白云山", "的", "海拔", "是", "多少", "?"]
    evidence_tokens = ["白云山", "坐落", "在", "广州", ",", "主峰", "海拔", "3", "8", "2", "米"]

    l = len(question_tokens)
    features = []

    for evidence_tmp in get_evidence_tmp(l, evidence_tokens):
        print(evidence_tmp)
        print(levenshtein(evidence_tmp, question_tokens))
        rate = round((levenshtein(evidence_tmp, question_tokens)) / l, 2)
        features.append(rate)
    print(features)
    return features

def test_jaccard():
    question_tokens = ["白云山", "的", "海拔", "是", "多少","?"]
    evidence_tokens = ["白云山", "坐落", "在","广州", ",", "主峰", "海拔", "3","8","2", "米"]

    l  = len(question_tokens)
    features = []

    for evidence_tmp in get_evidence_tmp(l, evidence_tokens):
        print(evidence_tmp)
        rate = cal_jaccard_similar_rate(evidence_tmp, question_tokens)
        features.append(rate)
    print (features)
    print(len(features))


def cal_jaccard_similar_rate(a_list, b_list):
    ret_list = list(set(a_list).union(set(b_list))) ## bing
    ret_list2 = list((set(a_list).union(set(b_list))) ^ (set(a_list) ^ set(b_list))) ## union

    return round((len(ret_list2)) / len(ret_list), 2)


def feature_extractor(filepath, output):
    f = open(filepath)
    result = open(output, 'w')
    for line in f:
        line = line.strip()
        linedict = json.loads(line)
        question_tokens = linedict['question_tokens']
        evidencelist = linedict['evidences']

        ## temp_dict for judge ee

        token_dict = {}
        for i, evidencedict in enumerate(evidencelist):
            evidence_tokens = evidencedict['evidence_tokens']
            for token in evidence_tokens:
                if token not in token_dict:
                    token_dict[token] = [i]
                elif i not in token_dict[token]:
                    token_dict[token].append(i)

        all_count = len(evidencelist)
        for k, v in token_dict.items():
            token_dict[k] = round(len(v)/all_count, 2)

        for evidencedict in evidencelist:
            evidence_ekey = evidencedict['e_key']
            evidence_tokens = evidencedict['evidence_tokens']

            ## add eefeature
            eefeature = []
            for token in evidence_tokens:
                eefeature.append(token_dict[token])
            evidencedict['f_eecomm'] = eefeature

            ## add_edit_distance
            feature_jasscard, feature_edit_distance = distance_features(question_tokens, evidence_tokens)
            evidencedict['f_edit_dist'] = feature_edit_distance
            evidencedict['f_jaccard'] = feature_jasscard

        outputline = json.dumps(linedict)
        result.write(outputline + "\n")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()


    parser.add_argument('--input_file', dest='input_path', type=str, help='filepath for the input data')
    parser.add_argument('--output_file', dest='output_path', type=str, help='filepath for the output data')
    args = parser.parse_args()
    input_path = args.input_path
    output_path = args.output_path

    # filename = "/media/iscas/linux/fym/data/sample.test"
    # output = "/media/iscas/linux/fym/data/sample.test_final"
    feature_extractor(input_path, output_path)

    ## python preprocess_post.py --input_file=/media/iscas/linux/fym/data/sample.test --output_file=/media/iscas/linux/fym/data/sample.test_final
#    test_jaccard()
#    test_editdistance()

