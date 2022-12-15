import myutils
import copy
from movie_reviews import makeData

class MyNaiveBayesClassifier:
    """
    Attributes:
        priors: The prior probabilities computed for each label in the training set.
        posteriors: The posterior probabilities computed for each attribute value/label pair in the training set.
    """
    def __init__(self):
        """Initializer for MyNaiveBayesClassifier.
        """
        self.priors = None
        self.posteriors = None

    def fit(self, X_train, y_train):
        """Fits a Naive Bayes classifier to X_train and y_train.

        Args:
            X_train(list of list of obj): The list of training instances (samples)
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples
        """
        x, y = myutils.groupby(X_train, y_train)
        
        # get all atts for each column
        cols = myutils.get_cols(X_train)
        tot_att_list = []
        for col in cols:
            a = []
            for item in col:
                if item not in a:
                    a.append(item)
            a.sort()
            tot_att_list.append(a)

        # priors
        priors = {}
        i=0
        for item in y:
            priors[item] = len(x[i])/len(X_train)
            i+=1

        self.priors = priors

        # posteriors
        nb = {}
        
        atts_list = []
        atts={}
        for i in range(len(X_train[0])):
            atts[i] = {}
        for i in range(len(y)):
            atts_list.append(copy.deepcopy(atts))

        i=0 
        for item in y:
            nb[item] = atts_list[i]
            try:
                atts_list[i+1]
                i+=1
            except:
                i=0

        cols = myutils.get_cols(X_train)

        freqs = []
        for item in cols:
            values, counts = myutils.get_frequencies(item)
            freqs.append(values)

        tables = []
        for table in x:
            s = []
            for item in table:
                s.append(X_train[item])
            tables.append(s)

        grouped_cols = []
        for item in tables:
            new_cols = myutils.get_cols(item)
            grouped_cols.append(new_cols)

        probs = []
        for table in grouped_cols:
            j = 0
            for col in table:
                col_values, col_counts = myutils.get_frequencies(col)
                p=0
                for i in range(len(tot_att_list[j])):
                    if tot_att_list[j][i] not in col_values:
                        probs.append(1 / (len(col_values)+len(tot_att_list)))
                    else:
                        probs.append((1+col_counts[p]) / (len(col_values)+len(tot_att_list)))
                        p+=1
                j+=1

        k=0
        for item in y:
            for i in range(len(X_train[0])):
                for j in range(len(freqs[i])):
                    d = {}
                    d[freqs[i][j]] = probs[k]
                    nb[item][i].update(d)
                    k+=1
                
        self.posteriors = nb

    def predict(self, X_test):
        """Makes predictions for test instances in X_test.
        Args:
            X_test(list of list of obj): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)
        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        y_predicted = []

        for instance in X_test:
            percentage_vals = copy.deepcopy(self.priors)
            for key, value in percentage_vals.items():
                val = percentage_vals[key]
                for i in range(len(instance)):
                    try:
                        val *= self.posteriors[key][0][instance[i]]
                    except:
                        # if word is not in either dict
                        pass
                percentage_vals[key] = val
            y_predicted.append(max(percentage_vals, key=percentage_vals.get))
        
        return y_predicted

    def create_martix(truth, senti_array):
        truth_arr = []
        false_arr = []

        if truth == '+':
            for ch in senti_array:
                if ch == '+':
                    truth_arr.append('+')
                if ch == '-':
                    false_arr.append('-')
        else:
            for ch in senti_array:
                if ch == '+':
                    false_arr.append('+')
                if ch == '-':
                    truth_arr.append('-')
        return truth_arr, false_arr

    def count_prediction(self, senti_array):
        pos = 0
        neg = 0
        for ch in senti_array:
            if ch == '+':
                pos+=1
            if ch == '-':
                neg+=1
        
        return pos, neg

    def get_pos_skew():
        file_neg = open("movie_reviews/neg.txt","r")
        neg_dict_count = len(file_neg.readlines())

        file_pos = open("movie_reviews/pos.txt","r")
        pos_dict_count = len(file_pos.readlines())

        pos_skew = pos_dict_count/neg_dict_count
        return pos_skew

    def posTestPredict(self):
        #POSITIVE TEST BAG PREDICTION
        posTestBag, negTestBag = makeData.main()
        i = 1
        overall_neg = 0
        overall_pos = 0
        for review in posTestBag:
            predicted = self.predict(review)
            pos, neg = self.count_prediction(predicted)

            skew = self.get_pos_skew()
            if pos > neg*skew:
                overall_pos+=1
            else:
                overall_neg+=1
            
        return overall_pos, overall_neg

    def negTestPredict(self):
        #NEGATIVE TEST BAG PREDICTION
        posTestBag, negTestBag = makeData.main()
        i = 1
        overall_neg = 0
        overall_pos = 0
        for review in negTestBag:
            predicted = self.predict(review)
            pos, neg = self.count_prediction(predicted)

            skew = self.get_pos_skew()
            if pos > neg*skew:
                overall_pos+=1
            else:
                overall_neg+=1
            
        return overall_pos, overall_neg

    def makeConfusionMatrix(self):
        actual = []
        predicted = []

        # true_pos, false_neg = self.posTestPredict()
        # actual.append(true_pos)
        # predicted.append(false_neg)

        # false_pos, true_neg = self.negTestPredict()
        # actual.append(false_pos)
        # predicted.append(true_neg)


        confusion_matrix = metrics.confusion_matrix(actual, predicted)
        cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = ["Positive", "Negative"])

        cm_display.plot()
        plt.show()