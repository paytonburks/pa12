from nb import MyNaiveBayesClassifier

def make_table(posname, negname):
    pos = open(posname).read().splitlines()
    neg = open(negname).read().splitlines()

    for i in range(len(pos)):
        pos[i] = [pos[i]]
        pos[i].append('+')
    for i in range(len(neg)):
        neg[i] = [neg[i]]
        neg[i].append('-')

    data = pos+neg
    return data

def main():
    movie_nb = MyNaiveBayesClassifier()
    data = make_table('movie_reviews/pos.txt', 'movie_reviews/neg.txt')

    X = []
    y = []

    for i in range(len(data)):
        X.append([data[i][0]])
        y.append(data[i][1])

    movie_nb.fit(X, y)
    print(movie_nb.posteriors)
    print(movie_nb.priors)

    return


main()
