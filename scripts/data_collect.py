#### ce code n'est pas encore testé et c'est à revoir.

"""
In this code, we will collect datasets that are in json format.
class Subforum contains two functions:
    _init_ function: to check the existence of zip files. # à revoir!!!!!!!!!
    _unzip_and_load function: to unzip and load 'zip files' which contain the data.    
"""

import os, re, sys
import nltk, json, codecs
import pydoc, math
import zipfile, random, datetime
#import itertools                   
#from operator import truediv
#from scipy.misc import comb
#from random import randrange
#from HTMLParser import HTMLParser


# Written by Doris Hoogeveen Nov 2015. For a usage please call the script without arguments.

def load_subforum(subforumzipped):
    ''' Takes a subforum.zip file as input and returns a StackExchange Subforum class object.'''
    return Subforum(subforumzipped)


class Subforum():
    def __init__(self, zipped_catfile):
        ''' This class takes a StackExchange subforum.zip file as input and makes it queryable via the 
            methods below. '''
        # Check to see if supplied file exists and is a valid zip file.(verification)
        if not os.path.exists(zipped_catfile):
            sys.exit(
             'The supplied zipfile does not exist. Please supply a valid StackExchange subforum.zip file.')
        if not zipfile.is_zipfile(zipped_catfile):
            sys.exit('Please supply a valid StackExchange subforum.zip file.')

        self.cat = os.path.basename(zipped_catfile).split('.')[0]  ### ?
        self._unzip_and_load(zipped_catfile) #_unzip_and_load: function implemented below.

        # Stopwords for cleaning. They need to be initialised here in case someone accesses self.stopwords.

        self.__indri_stopwords = ['a', 'about', 'above', 'according', 'across',
                                  'after', 'afterwards', 'again', 'against',
                                  'albeit', 'all', 'almost', 'alone', 'along',
                                  'already', 'also', 'although', 'always',
                                  'am', 'among', 'amongst', 'an', 'and',
                                  'another', 'any', 'anybody', 'anyhow',
                                  'anyone', 'anything', 'anyway', 'anywhere',
                                  'apart', 'are', 'around', 'as', 'at', 'av',
                                  'be', 'became', 'because', 'become',
                                  'becomes', 'becoming', 'been', 'before',
                                  'beforehand', 'behind', 'being', 'below',
                                  'beside', 'besides', 'between', 'beyond',
                                  'both', 'but', 'by', 'can', 'cannot',
                                  'canst', 'certain', 'cf', 'choose',
                                  'contrariwise', 'cos', 'could', 'cu', 'day',
                                  'do', 'does', "doesn't", 'doing', 'dost',
                                  'doth', 'double', 'down', 'dual', 'during',
                                  'each', 'either', 'else', 'elsewhere',
                                  'enough', 'et', 'etc', 'even', 'ever',
                                  'every', 'everybody', 'everyone',
                                  'everything', 'everywhere', 'except',
                                  'excepted', 'excepting', 'exception',
                                  'exclude', 'excluding', 'exclusive', 'far',
                                  'farther', 'farthest', 'few', 'ff', 'first',
                                  'for', 'formerly', 'forth', 'forward',
                                  'from', 'front', 'further', 'furthermore',
                                  'furthest', 'get', 'go', 'had', 'halves',
                                  'hardly', 'has', 'hast', 'hath', 'have',
                                  'he', 'hence', 'henceforth', 'her', 'here',
                                  'hereabouts', 'hereafter', 'hereby',
                                  'herein', 'hereto', 'hereupon', 'hers',
                                  'herself', 'him', 'himself', 'hindmost',
                                  'his', 'hither', 'hitherto', 'how',
                                  'however', 'howsoever', 'i', 'ie', 'if',
                                  'in', 'inasmuch', 'inc', 'include',
                                  'included', 'including', 'indeed', 'indoors',
                                  'inside', 'insomuch', 'instead', 'into',
                                  'inward', 'inwards', 'is', 'it', 'its',
                                  'itself', 'just', 'kind', 'kg', 'km', 'last',
                                  'latter', 'latterly', 'less', 'lest', 'let',
                                  'like', 'little', 'ltd', 'many', 'may',
                                  'maybe', 'me', 'meantime', 'meanwhile',
                                  'might', 'moreover', 'most', 'mostly',
                                  'more', 'mr', 'mrs', 'ms', 'much', 'must',
                                  'my', 'myself', 'namely', 'need', 'neither',
                                  'never', 'nevertheless', 'next', 'no',
                                  'nobody', 'none', 'nonetheless', 'noone',
                                  'nope', 'nor', 'not', 'nothing',
                                  'notwithstanding', 'now', 'nowadays',
                                  'nowhere', 'of', 'off', 'often', 'ok', 'on',
                                  'once', 'one', 'only', 'onto', 'or', 'other',
                                  'others', 'otherwise', 'ought', 'our',
                                  'ours', 'ourselves', 'out', 'outside',
                                  'over', 'own', 'per', 'perhaps', 'plenty',
                                  'provide', 'quite', 'rather', 'really',
                                  'round', 'said', 'sake', 'same', 'sang',
                                  'save', 'saw', 'see', 'seeing', 'seem',
                                  'seemed', 'seeming', 'seems', 'seen',
                                  'seldom', 'selves', 'sent', 'several',
                                  'shalt', 'she', 'should', 'shown',
                                  'sideways', 'since', 'slept', 'slew',
                                  'slung', 'slunk', 'smote', 'so', 'some',
                                  'somebody', 'somehow', 'someone',
                                  'something', 'sometime', 'sometimes',
                                  'somewhat', 'somewhere', 'spake', 'spat',
                                  'spoke', 'spoken', 'sprang', 'sprung',
                                  'stave', 'staves', 'still', 'such',
                                  'supposing', 'than', 'that', 'the', 'thee',
                                  'their', 'them', 'themselves', 'then',
                                  'thence', 'thenceforth', 'there',
                                  'thereabout', 'thereabouts', 'thereafter',
                                  'thereby', 'therefore', 'therein', 'thereof',
                                  'thereon', 'thereto', 'thereupon', 'these',
                                  'they', 'this', 'those', 'thou', 'though',
                                  'thrice', 'through', 'throughout', 'thru',
                                  'thus', 'thy', 'thyself', 'till', 'to',
                                  'together', 'too', 'toward', 'towards',
                                  'ugh', 'unable', 'under', 'underneath',
                                  'unless', 'unlike', 'until', 'up', 'upon',
                                  'upward', 'upwards', 'us', 'use', 'used',
                                  'using', 'very', 'via', 'vs', 'want', 'was',
                                  'we', 'week', 'well', 'were', 'what',
                                  'whatever', 'whatsoever', 'when', 'whence',
                                  'whenever', 'whensoever', 'where',
                                  'whereabouts', 'whereafter', 'whereas',
                                  'whereat', 'whereby', 'wherefore',
                                  'wherefrom', 'wherein', 'whereinto',
                                  'whereof', 'whereon', 'wheresoever',
                                  'whereto', 'whereunto', 'whereupon',
                                  'wherever', 'wherewith', 'whether', 'whew',
                                  'which', 'whichever', 'whichsoever', 'while',
                                  'whilst', 'whither', 'who', 'whoa',
                                  'whoever', 'whole', 'whom', 'whomever',
                                  'whomsoever', 'whose', 'whosoever', 'why',
                                  'will', 'wilt', 'with', 'within', 'without',
                                  'worse', 'worst', 'would', 'wow', 'ye',
                                  'yet', 'year', 'yippee', 'you', 'your',
                                  'yours', 'yourself', 'yourselves']
        self.__short_stopwords = ['a', 'an', 'the', 'yes', 'no', 'thanks']
        self.__middle_stopwords = ['in', 'on', 'at', 'a', 'an', 'is', 'be',
                                   'was', 'I', 'you', 'the', 'do', 'did', 'of',
                                   'so', 'for', 'with', 'yes', 'thanks']

        # The NLTK stopwords cause a problem if they have not been downloaded, so we need a check for that.
        try:
            self.__nltk_stopwords = nltk.corpus.stopwords.words('english')
        except:
            self.__nltk_stopwords = []

        self.__stopwords = self.__middle_stopwords  # Default.
        self.cutoffdate = False  # Needed for classification splits.

    def _unzip_and_load(self, zipped_catfile):
        ziplocation = os.path.dirname(zipped_catfile)
        cat = os.path.basename(zipped_catfile).split('.')[0]
        questionfile = ziplocation + '/' + cat + '/' + cat + '_questions.json'
        answerfile = ziplocation + '/' + cat + '/' + cat + '_answers.json'
        commentfile = ziplocation + '/' + cat + '/' + cat + '_comments.json'
        userfile = ziplocation + '/' + cat + '/' + cat + '_users.json'
        if os.path.exists(questionfile) and os.path.exists(                 #verifaction
                answerfile) and os.path.exists(userfile) and os.path.exists(
                commentfile):
            pass  # All good, we don't need to unzip anything
        else:
            zip_ref = zipfile.ZipFile(zipped_catfile, 'r')
            zip_ref.extractall(ziplocation)

        qf = codecs.open(questionfile, 'r', encoding='utf-8') #loaded file in json format
        self.postdict = json.load(qf)

        af = codecs.open(answerfile, 'r', encoding='utf-8')
        self.answerdict = json.load(af)

        cf = codecs.open(commentfile, 'r', encoding='utf-8')
        self.commentdict = json.load(cf)

        uf = codecs.open(userfile, 'r', encoding='utf-8')
        self.userdict = json.load(uf)

        print
        "Loaded all data from", zipped_catfile

#    def tokenize(self, s):
#        ''' Takes a string as input, tokenizes it using NLTK (http://www.nltk.org) and returns a list of the tokens. '''
#        return nltk.word_tokenize(
#            s)  # The NLTK tokenizer cuts things like 'cannot' into 'can' and 'not'.
