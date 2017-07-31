import os
import copy

import numpy as np
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import json
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from sklearn.datasets import make_multilabel_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import zero_one_loss

from libact.base.dataset import Dataset
from libact.labelers import IdealLabeler
from libact.query_strategies import RandomSampling
from libact.query_strategies.multilabel import MultilabelWithAuxiliaryLearner
from libact.query_strategies.multilabel import MMC, BinaryMinimization
from libact.models.multilabel import BinaryRelevance
from libact.models import LogisticRegression, SVM


def remove_stop_words(sentence):
    """ Removes stopwords from full_sentences.
    Parameters
    ----------
    sentence: list
        List of tokenized words.
    Returns
    -------
    filtered_sent : list
        New sentence with stopwords removed.
    """
    stop_words = stopwords.words('english')
    stop_words.extend((',', '.'))  # Removing punctuation increases accuracy.

    filtered_sent = [w for w in sentence if w.lower() not in stop_words]
    return filtered_sent


def load_glove_model(gloveFile, word_count=100000):
    """
    Loads a selection of the most common glove vectors into a dictionary.
    :param gloveFile: textfile
        Contains glove vectors. Each line contains a word string followed by a
        'n_features'-dimensional vector to describe the word. Where n_features
        is the number of features.
    :param word_count: int, default: 100000
        Number of words to load from the gloveFile
    :return: dictionary
        {'word': vector}
        word = string of the word we wish to load
        vector = 'n_features'-d vectors to describe the word
    """
    path = './data/gloveFile_done_%d.npy' % (word_count)

    # Saves time by loading existing file, if available.
    if os.path.exists(path):
        glove = np.load(path).item()
        print '100,000 glove vectors loaded from existing file.'

    else:
        f = open(gloveFile,'r')
        glove = {}
        count = 0
        for line in f:
            split_line = line.split()
            word = split_line[0]
            embedding = [float(val) for val in split_line[1:]]
            embedding = np.asanyarray(embedding)
            glove[word] = embedding
            count += 1
            if count >= word_count:
                break
        # Saves the vectors, so we can load it faster next time.
        np.save(path, glove)
        print '100,000 glove vectors loaded.'

    return glove

def load_data_mal(model, loc='./data/maluuba/data_frames.json'):
    """Loads data directly from JSON. JSON dataset must contain the keys
    "labels" and "dialog_list", for example:
    "labels": [["inform"], ["suggest"], ["negate", "thankyou"]],
    "dialog_list": [
        "HI im from Alexandria looking to book a 12 day vacation",
        "I only have flights available to Vancouver, Canada at the moment.",
        "I will look elsewhere thank you!"]

    :param loc: string
        Locates the directory in which the data is saved.

    :return: X, numpy array shape (n_samples, n_features)

    :return: Y, numpy array shape (n_samples, n_classes)
    """
    X_list, Y_list = [], []
    
    with open(loc, 'r') as file:
        for line in file:
            line = json.loads(line)
            dialog_list = line.get('dialog_list', 'empty')
            labels = line.get('labels', 'empty')
            
            # Since, for now, each dialog is an example i.e. we ignore context
            X_list.extend(dialog_list)

            # All labels in Y_list correspond to each dialog turn as embedded
            # lists. Example:
            # [['label_7'],['label_2'], ['label_4','label_1'], ['label_3'], ..]
            Y_list.extend(labels)

    # Y is a numpy array, shape (n_samples, n_classes)
    mlb = MultiLabelBinarizer()
    Y = mlb.fit_transform(Y_list)


    # Place each example (vector representation of a turn in conversation),
    # as a row in a matrix.  X has shape (n_examples, n_features)
    X = np.asarray(sentences2vec(X_list, model))

    return X, Y


def load_data_amz(model, loc='./data/amazon/amazon_mixed_dataset.json'):
    """Loads data directly from JSON. Input data should have the form:
        ...what about stop words???

    :param loc: string
        Locates the directory in which the data is saved.

    :return: X, numpy array shape (n_samples, n_classes)

    :return: Y, numpy array shape (n_samples, n_classes)
    """
    X_list, Y_list = [], []

    with open(loc, 'r') as file:
        for line in file:
            line = json.loads(line)
            review = line.get('text', 'empty')
            labels = line.get('labels', 'empty')

            # Since, for now, each dialog is an example i.e. we ignore context
            X_list.extend([review])

            # All labels in Y_list correspond to each dialog turn as embedded
            # lists. Example:
            # [['label_7'],['label_2'], ['label_4','label_1'], ['label_3'], ..]
            Y_list.extend([[labels]])

    # Y is a numpy array, shape (n_samples, n_classes)
    mlb = MultiLabelBinarizer()
    Y = mlb.fit_transform(Y_list)

    # Place each example (vector representation of a turn in conversation),
    # as a row in a matrix.  X has shape (n_examples, n_features)
    X = np.asarray(sentences2vec(X_list, model))

    return X, Y

def sentences2vec(sentences, glove):
    """ Uses the glove word vectors to convert whole sentences to single
    vectors by averaging the input vectors.
    
    :param: sentences, list
        List of all the dialog turns as a list of strings. 
        Example:
        ['First conversation first turn.','First conversation second turn.',
         ..., 'Last conversation, last turn']
    :param glove: dictionary
        {'word': vector}, where vector has shape, (1, 'n_features')

    :return: list
        List of sentences, where each sentence is represented by a
        single vector.
    """
    if type(sentences) != list: 
        print "sentences2vec requires a list of sentence/s"
        return

    sentence_vectors = []
    for sentence in sentences:
        sentence = word_tokenize(sentence)

        # Optional: remove stop words
        # stop_removed = remove_stop_words(sentence)
        # If, when we remove stopwords, turns of the conversation have no
        # content, we leave the stopwords in
        # if len(stop_removed )!=0: sentence = stop_removed

        matched_words = len(sentence)

        # All vectors have the same dimensionality. Using an example
        # 'word' to set the size of our new sentence vector.
        sum_of_words = np.zeros(len(glove[',']))
        
        # We will represent a sentence as an average of the words it contains.
        for word in sentence:
            sum_of_words += glove.get(word, 0)

        if matched_words != 0: # Necessary, to ensure we don't divide by zero.
            sentence_vector = sum_of_words/matched_words
            sentence_vectors.append(sentence_vector)
        else:
            # For empty reviews we will use a placeholder
            # (will change code so this entry is deleted from dataset)
            sentence_vectors.append(glove.get(',', 0))

    return sentence_vectors


def full_trn_tst(X, Y, train_size, test_size, num_labelled):

    # Preprocessing:
    # X = StandardScaler().fit_transform(X)
    # if args.pca: X = process_pca(args.pca, X)
    # else: pass

    # Randomly split the X and y arrays according to test_size.
    X_trn, X_tst, Y_trn, Y_tst = train_test_split(X, Y, test_size=test_size,
                                                  train_size=train_size)

    # We are giving the Dataset 'num_labelled' fully labelled examples 
    Y_trn_mix = Y_trn[:num_labelled].tolist()+[None]*(len(Y_trn)-num_labelled)
    
    # Dataset is a libact object that stores labelled and unlabelled data 
    # (unlabelled examples are stored as None)
    trn_ds = Dataset(X_trn, Y_trn_mix)
    tst_ds = Dataset(X_tst, Y_tst.tolist())
    fully_labeled_trn_ds = Dataset(X_trn, Y_trn)

    return trn_ds, tst_ds, fully_labeled_trn_ds


def split_train_test(test_size, num_labelled):
    """
    Generates synthetic data for multilabel problem.

    :param test_size:
    :param num_labelled:
    :return:
    """
    # choose a dataset with unbalanced class instances
    # calls random sample generators to build artificial datasets
    # data[0] are the examples; data[1] are the labels
    data = make_multilabel_classification(
        n_samples=300, n_classes=10, allow_unlabeled=False)

    # Standardize features by removing the mean and scaling to unit variance
    X = StandardScaler().fit_transform(data[0])
    Y = data[1]

    # Split X & Y into random train and test subsets
    X_trn, X_tst, Y_trn, Y_tst = train_test_split(X, Y, test_size=test_size)

    # We are giving the Dataset 'num_labelled' fully labelled examples
    Y_trn_mix = Y_trn[:num_labelled].tolist() + [None] * (
    len(Y_trn) - num_labelled)

    # Dataset is a libact object that stores labelled and unlabelled data
    # (unlabelled examples are stored as None)
    trn_ds = Dataset(X_trn, Y_trn_mix)
    tst_ds = Dataset(X_tst, Y_tst.tolist())

    fully_labeled_trn_ds = Dataset(X_trn, Y_trn)

    return trn_ds, tst_ds, fully_labeled_trn_ds


def run(trn_ds, tst_ds, lbr, model, qs, quota, fully_labeled_trn_ds):
    C_out, C_out_01 = [], []

    # Used to produce additional metrics not covered in libact
    tst_X, tst_y = zip(*tst_ds.data)

    for _ in range(quota):
        # Query strategy (MMC, BinMin, RandomSampling,
        # MultilabelWithAuxiliaryLearner) returns id of example to query
        ask_id = qs.make_query()

        # Returns the example corresponding to ask_id
        X, _ = trn_ds.data[ask_id]
        # Simulated oracle returns the label for the example, x
        lb = lbr.label(X)

        # Update training library with new label
        trn_ds.update(ask_id, lb)

        # Train the model (usually Binary Relevance) with the additional label
        model.train(trn_ds)

        # score returns the mean accuracy on the test dataset. In this case we
        # have chosen to recieve the Hamming loss, which is the fraction of
        # labels that are incorrectly predicted
        C_out = np.append(C_out, model.score(tst_ds, criterion='hamming'))
        C_out_01 = np.append(C_out_01, zero_one_loss(np.asarray(tst_y), model.predict(tst_X)))


    return C_out, C_out_01


def experiments(fully_labeled_trn_ds, trn_ds, tst_ds, quota):
    """
    Runs the active learning experiment for each of the 6 query
    strategies/criteria combinations.

    :param fully_labeled_trn_ds: libact dataset
        Fully labelled training examples
    :param trn_ds: libact dataset
        Training examples with missing labels
    :param tst_ds: libact dataset
        Test dataset
    :param quota: int
        number of queries
    :return: results: dict
        Key describes the type of experiment, while the value is the
        corresponding result from the loss function. The value is a list which
        tracks the Hamming loss and f1 score, in positions 0 and 1. The Hamming
        loss and f1 score themselves are lists corresponding to the loss after
        making different amounts of queries.
    """
    # lbr simulates an oracle (method label(x) returns x's label)
    lbr = IdealLabeler(fully_labeled_trn_ds)

    # This is the model which we train and use for predictions. It divides the
    # task into multiple binary tasks, training one classifier for each label.
    # Unlike OvA and OvR BinaryRelevance does not use one classifier for each
    # possible value for the label
    model = BinaryRelevance(LogisticRegression())

    trn_ds2 = copy.deepcopy(trn_ds)
    trn_ds3 = copy.deepcopy(trn_ds)
    trn_ds4 = copy.deepcopy(trn_ds)
    trn_ds5 = copy.deepcopy(trn_ds)
    trn_ds6 = copy.deepcopy(trn_ds)

    results = {}

    print "before qs1"
    # MMC requires a 'base learner' for its binary relevance
    qs = MMC(trn_ds, br_base=LogisticRegression())
    E_out_1, E_out_01_1 = run(trn_ds, tst_ds, lbr, model, qs, quota,
                              fully_labeled_trn_ds)
    results['MMC'] = [E_out_1, E_out_01_1]

    print "before qs2"

    qs2 = RandomSampling(trn_ds2)
    E_out_2, E_out_01_2 = run(trn_ds2, tst_ds, lbr, model, qs2, quota,
                              fully_labeled_trn_ds)
    results['Random'] = [E_out_2, E_out_01_2]

    print "before qs3"

    # MLALAL requires two multilabel models:
    # 1) major_learner:
    #    Gives a binary output on each label and is used for final prediction
    # 2) auxiliary_learner:
    #    Estimates the confident on each label, and gives a real value output
    #   (supports pred_real method)

    qs3 = MultilabelWithAuxiliaryLearner(
        trn_ds3,
        BinaryRelevance(LogisticRegression()),
        BinaryRelevance(SVM()),
        criterion='hlr')
    E_out_3, E_out_01_3 = run(trn_ds3, tst_ds, lbr, model, qs3, quota,
                              fully_labeled_trn_ds)
    results['Aux_hlr'] = [E_out_3, E_out_01_3]

    print "before qs4"

    qs4 = MultilabelWithAuxiliaryLearner(
        trn_ds4,
        BinaryRelevance(LogisticRegression()),
        BinaryRelevance(SVM()),
        criterion='shlr')
    E_out_4, E_out_01_4 = run(trn_ds4, tst_ds, lbr, model, qs4, quota,
                              fully_labeled_trn_ds)
    results['Aux_shlr'] = [E_out_4, E_out_01_4]

    print "before qs5"

    qs5 = MultilabelWithAuxiliaryLearner(
        trn_ds5,
        BinaryRelevance(LogisticRegression()),
        BinaryRelevance(SVM()),
        criterion='mmr')
    E_out_5, E_out_01_5 = run(trn_ds5, tst_ds, lbr, model, qs5, quota,
                              fully_labeled_trn_ds)
    results['Aux_mmr'] = [E_out_5, E_out_01_5]

    print "before qs6"
    # BinMin only needs a Continuous Model to eveluate uncertainty
    qs6 = BinaryMinimization(trn_ds6, LogisticRegression())
    E_out_6, E_out_01_6 = run(trn_ds6, tst_ds, lbr, model, qs6, quota,
                              fully_labeled_trn_ds)
    results['BinMin'] = [E_out_6, E_out_01_6]

    print "Done qs6"

    return results

###############################################################################
##### function used for creating amazon dataset ###############################
#
# import json
# import gzip
# import os
# import glob
#
# def parse(path):
#     g = gzip.open(path, 'r')
#     for l in g:
#         yield eval(l)
#
# with open('./data/amazon/amazon_mixed_dataset.json', 'w') as outfile:
#     path = "./data/amazon/*.gz"
#     for fname in glob.glob(path):
#         i = 0
#         for line in parse(fname):
#             newline = {}
#             # Should be a better way to do this?
#             newline['labels'] = category = \
#             os.path.basename(fname).split('.')[0][8:]
#             newline['text'] = line['reviewText']
#             json.dump(newline, outfile)
#             outfile.write('\n')
#             i += 1
#             if i >= 1000: break
###############################################################################
###############################################################################
