__author__ = 'bernhard'

import os
import logging
import datetime
import re

log = logging.Logger(__name__)
log.setLevel(logging.DEBUG)


class ExtractionCache(object):
    __ENDINGS = {
        'wiki': '.wiki',
        'e2rdf': '.e2rdf',
        'coref': '.coref',
        'openie': '.openie',
        'nt': '.nt'
    }

    __TIME_FORMAT = '%a %b %d %H:%M:%S %Y'

    def __init__(self, path):
        if os.path.isdir(path):
            log.info("Path for ExtractionCache exists at location: '{}'".format(path))
        else:
            log.warning("Path for ExtractionCache does not exist yet, creating it: '{}'".format(path))
            os.mkdir(path=path, mode=0o770)

        self.__counter = 0
        self.__hit_counter = 0
        self.__miss_counter = 0
        self.__out_of_date_counter = 0

        self.__paths = {}
        for p in self.__ENDINGS.keys():
            self.__paths[p] = path + os.sep + p + os.sep

        for (_, p) in self.__paths.items():
            if not os.path.isdir(p):
                os.mkdir(path=p, mode=0o770)

    def __get(self, article_name, filetype, max_days=90):
        filename = self.__paths[filetype] + self.__cleaned_article_name(article_name) + self.__ENDINGS[filetype]
        self.__counter += 1

        if os.path.isfile(filename):
            self.__hit_counter += 1

            with open(filename, mode='r', encoding='utf-8') as input_file:
                first_line = input_file.readline().rstrip()  # read only first line and remove newline character
                created = datetime.datetime.strptime(first_line, ExtractionCache.__TIME_FORMAT)

                if max_days > 0 and (created + datetime.timedelta(days=max_days) < datetime.datetime.now()):
                    self.__out_of_date_counter += 1
                    self.__hit_counter -= 1  # we hit, but content was out of date
                    raise self.CacheOutOfDateException(
                        message="Cache for the article '{}' is over {} days old, you must do another extraction".format(article_name, max_days))

                rest_of_file = input_file.read()
                return rest_of_file

        else:
            self.__miss_counter += 1
            raise self.CacheMissedException(message="Content for article '{}' is not in cache.".format(article_name))

    @staticmethod
    def __cleaned_article_name(article_name):
        return re.sub(r'\W', repl='-', string=article_name)

    def __put(self, article_name, content, filetype):
        filename = self.__paths[filetype] + self.__cleaned_article_name(article_name) + self.__ENDINGS[filetype]

        try:
            self.__delete(article_name, filetype)
        except ExtractionCache.CacheMissedException:
            pass

        with open(filename, mode='w', encoding='utf-8') as output_file:
            output_file.write("{}\n".format(datetime.datetime.now().strftime(ExtractionCache.__TIME_FORMAT)))
            output_file.write(content)

    def __delete(self, article_name, filetype):
        filename = self.__paths[filetype] + self.__cleaned_article_name(article_name) + self.__ENDINGS[filetype]
        if os.path.isfile(filename):
            os.remove(filename)
        else:
            raise self.CacheMissedException(
                message="Tried to delete wiki article '{}' but wasn't there.".format(article_name))

    def get_wiki(self, article_name):
        """
        Returns the wikipedia text for the given article_name if it is in the cache.
        Otherwise it raises a CacheMissedException.
        :param article_name: the name of the article which content to get
        :return: The stored content for the given article.
        """
        return self.__get(article_name, 'wiki')

    def put_wiki(self, article_name, article):
        """
        Stores the given article for the given article_name on the disk for later usage.
        Already existing files will be overwritten.
        :param article_name: name of the article
        :param article: content of the article
        """
        self.__put(article_name, article, 'wiki')

    def get_e2rdf(self, article_name):
        """
        Returns the cached extractions for the given article.
        :param article_name:
        :type article_name str
        :return:
        """
        return self.__get(article_name, 'e2rdf')

    def put_e2rdf(self, article_name, extraction):
        self.__put(article_name, extraction, 'e2rdf')

    def get_nt(self, article_name):
        return self.__get(article_name, 'nt')

    def put_nt(self, article_name, nt):
        self.__put(article_name, nt, 'nt')

    def get_coref(self, article_name):
        return self.__get(article_name, 'coref')

    def put_coref(self, article_name, coreferences):
        self.__put(article_name, coreferences, 'coref')

    def get_openie(self, article_name):
        return self.__get(article_name, 'openie')

    def put_openie(self, article_name, openie):
        self.__put(article_name, openie, 'openie')

    def get_statistics(self):
        c_articles = self.__count_stored(self.__paths['wiki'], self.__ENDINGS['wiki'])
        c_coref = self.__count_stored(self.__paths['coref'], self.__ENDINGS['coref'])
        c_openie = self.__count_stored(self.__paths['openie'], self.__ENDINGS['openie'])
        c_ext = self.__count_stored(self.__paths['e2rdf'], self.__ENDINGS['e2rdf'])
        return {
            'requests': "{}".format(self.__counter),
            'hits': "{}".format(self.__hit_counter),
            'misses': "{}".format(self.__miss_counter),
            'hit-rate': "{:.3}".format(
                0.0 if self.__counter == 0 else float(self.__hit_counter) / float(self.__counter)),
            'miss-rate': "{:.3}".format(
                0.0 if self.__counter == 0 else float(self.__miss_counter) / float(self.__counter)),
            'out-of-date': "{}".format(self.__out_of_date_counter),
            'article_count': "{}".format(c_articles),
            'coref_count': "{}".format(c_coref),
            'openie_count': "{}".format(c_openie),
            'ext_count': "{}".format(c_ext),
            'total_count': "{}".format(c_articles + c_coref + c_openie + c_ext)
        }

    @staticmethod
    def __count_stored(path, ending):
        return sum(1 for p in os.listdir(path) if p.endswith(ending))

    class CacheException(Exception):
        def __init__(self, message, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.message = message

        def __str__(self):
            return self.message

    class CacheMissedException(CacheException):
        def __init__(self, message, *args, **kwargs):
            super().__init__(message, *args, **kwargs)

    class CacheOutOfDateException(CacheMissedException):
        def __init__(self, message, *args, **kwargs):
            super().__init__(message, *args, **kwargs)
