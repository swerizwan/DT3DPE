# Import necessary libraries
import numpy as np
import pickle
from os.path import join as pjoin

# Define a dictionary to enumerate Part-of-Speech (POS) tags
POS_enumerator = {
    'VERB': 0,
    'NOUN': 1,
    'DET': 2,
    'ADP': 3,
    'NUM': 4,
    'AUX': 5,
    'PRON': 6,
    'ADJ': 7,
    'ADV': 8,
    'Loc_VIP': 9,
    'Body_VIP': 10,
    'Obj_VIP': 11,
    'Act_VIP': 12,
    'Desc_VIP': 13,
    'OTHER': 14,
}

# Define lists for specific categories of words
Loc_list = ('left', 'right', 'clockwise', 'counterclockwise', 'anticlockwise', 'forward', 'back', 'backward',
            'up', 'down', 'straight', 'curve')

Body_list = ('arm', 'chin', 'foot', 'feet', 'face', 'hand', 'mouth', 'leg', 'waist', 'eye', 'knee', 'shoulder', 'thigh')

Obj_List = ('stair', 'dumbbell', 'chair', 'window', 'floor', 'car', 'ball', 'handrail', 'baseball', 'basketball')

Act_list = ('walk', 'run', 'swing', 'pick', 'bring', 'kick', 'put', 'squat', 'throw', 'hop', 'dance', 'jump', 'turn',
            'stumble', 'dance', 'stop', 'sit', 'lift', 'lower', 'raise', 'wash', 'stand', 'kneel', 'stroll',
            'rub', 'bend', 'balance', 'flap', 'jog', 'shuffle', 'lean', 'rotate', 'spin', 'spread', 'climb')

Desc_list = ('slowly', 'carefully', 'fast', 'careful', 'slow', 'quickly', 'happy', 'angry', 'sad', 'happily',
             'angrily', 'sadly')

# Create a dictionary mapping VIP categories to their respective lists
VIP_dict = {
    'Loc_VIP': Loc_list,
    'Body_VIP': Body_list,
    'Obj_VIP': Obj_List,
    'Act_VIP': Act_list,
    'Desc_VIP': Desc_list,
}


# Define a class for word vectorization
class WordVectorizer(object):
    """
    A class for converting words to their vector representations and handling POS tags.
    """
    def __init__(self, meta_root, prefix):
        """
        Initializes the WordVectorizer with word vectors and indices.
        :param meta_root: The root directory containing the metadata files.
        :param prefix: The prefix for the metadata files.
        """
        vectors = np.load(pjoin(meta_root, '%s_data.npy' % prefix))
        words = pickle.load(open(pjoin(meta_root, '%s_words.pkl' % prefix), 'rb'))
        self.word2idx = pickle.load(open(pjoin(meta_root, '%s_idx.pkl' % prefix), 'rb'))
        self.word2vec = {w: vectors[self.word2idx[w]] for w in words}

    def _get_pos_ohot(self, pos):
        """
        Converts a POS tag to a one-hot encoded vector.
        :param pos: The POS tag.
        :return: A one-hot encoded vector for the POS tag.
        """
        pos_vec = np.zeros(len(POS_enumerator))
        if pos in POS_enumerator:
            pos_vec[POS_enumerator[pos]] = 1
        else:
            pos_vec[POS_enumerator['OTHER']] = 1
        return pos_vec

    def __len__(self):
        """
        Returns the number of words in the word2vec dictionary.
        :return: The number of words.
        """
        return len(self.word2vec)

    def __getitem__(self, item):
        """
        Retrieves the word vector and POS vector for a given word/POS pair.
        :param item: A string in the format 'word/POS'.
        :return: A tuple containing the word vector and POS vector.
        """
        word, pos = item.split('/')
        if word in self.word2vec:
            word_vec = self.word2vec[word]
            vip_pos = None
            for key, values in VIP_dict.items():
                if word in values:
                    vip_pos = key
                    break
            if vip_pos is not None:
                pos_vec = self._get_pos_ohot(vip_pos)
            else:
                pos_vec = self._get_pos_ohot(pos)
        else:
            word_vec = self.word2vec['unk']
            pos_vec = self._get_pos_ohot('OTHER')
        return word_vec, pos_vec


# Define a subclass for an extended version of WordVectorizer
class WordVectorizerV2(WordVectorizer):
    """
    An extended version of WordVectorizer with additional functionality.
    """
    def __init__(self, meta_root, prefix):
        """
        Initializes the WordVectorizerV2 with word vectors and indices.
        :param meta_root: The root directory containing the metadata files.
        :param prefix: The prefix for the metadata files.
        """
        super(WordVectorizerV2, self).__init__(meta_root, prefix)
        self.idx2word = {self.word2idx[w]: w for w in self.word2idx}

    def __getitem__(self, item):
        """
        Retrieves the word vector, POS vector, and word index for a given word/POS pair.
        :param item: A string in the format 'word/POS'.
        :return: A tuple containing the word vector, POS vector, and word index.
        """
        word_vec, pose_vec = super(WordVectorizerV2, self).__getitem__(item)
        word, pos = item.split('/')
        if word in self.word2vec:
            return word_vec, pose_vec, self.word2idx[word]
        else:
            return word_vec, pose_vec, self.word2idx['unk']

    def itos(self, idx):
        """
        Converts an index to its corresponding word.
        :param idx: The index to convert.
        :return: The word corresponding to the index.
        """
        if idx == len(self.idx2word):
            return "pad"
        return self.idx2word[idx]