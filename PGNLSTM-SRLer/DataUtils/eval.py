# refer to https://github.com/bcmi220/unisrl/blob/master/srl_eval_utils.py & https://github.com/RuiCaiNLP/Semi_SRL/blob/master/nnet/run/srl/util.py

class Eval:
    def __init__(self):
        self.predict_num = 0
        self.correct_num = 0
        self.gold_num = 0

        self.precision = 0
        self.recall = 0
        self.fscore = 0

        self.acc = 0

    def clear_PRF(self):
        self.predict_num = 0
        self.correct_num = 0
        self.gold_num = 0

        self.precision = 0
        self.recall = 0
        self.fscore = 0

    def getFscore(self, y_pred, y_true, all_sentence_length):

        for i in range(len(y_true)):
            sentence_length = all_sentence_length[i]

            for g_lable in y_true[i][:sentence_length]:
                for p_lable in y_pred[i][:sentence_length]:
                    if (p_lable == g_lable) and (p_lable!= '_'): self.correct_num += 1

            true_labels = [item for item in y_true[i][:sentence_length] if item != '_']
            pred_labels = [item for item in y_pred[i][:sentence_length] if item != '_']
            self.predict_num += len(pred_labels)
            self.gold_num += len(true_labels)

        if self.predict_num == 0:
            self.precision = 0
        else:
            self.precision = (self.correct_num / self.predict_num) * 100

        if self.gold_num == 0:
            self.recall = 0
        else:
            self.recall = (self.correct_num / self.gold_num) * 100

        if self.precision + self.recall == 0:
            self.fscore = 0
        else:
            self.fscore = (2 * (self.precision * self.recall)) / (self.precision + self.recall)

        self.accuracy(y_pred, y_true, all_sentence_length)

        return self.precision, self.recall, self.fscore, self.acc

    def accuracy(self, predict_labels, gold_labels, all_sentence_length): 
        cor = 0
        totol_leng = sum([len(predict_label) for predict_label in predict_labels])

        for p_lable, g_lable, sentence_length in zip(predict_labels, gold_labels, all_sentence_length):
            for p_lable_, g_lable_ in zip(p_lable[:sentence_length], g_lable[:sentence_length]):
                if p_lable_ == g_lable_:
                    cor += 1

        self.acc = cor / totol_leng * 100

        return self.acc
