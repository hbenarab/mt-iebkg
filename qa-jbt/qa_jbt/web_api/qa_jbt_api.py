__author__ = 'bernhard'

import os
import hashlib
import requests as req

from flask_restful import Resource, Api

from qa_jbt.cache.ExtractionCache import ExtractionCache

DATA_PATH = os.path.normpath(os.path.join(os.path.split(os.path.abspath(__file__))[0], '..', '..', 'data'))
MAX_MEMORY = 10  # (in GB) max memory to use for coref and openie for each call

import wikipedia_wrapper.get_article as wiki  # module to get wikipedia content
from jbt_berkeley_coref_resolution import coref  # module to to coreference resolution
from openie import openie
import ext2rdf.e2rdf as e2rdf


class JbtApi(Api):
    def __init__(self, app):
        super().__init__(app)
        self.__add_resources()

    def __add_resources(self):
        self.add_resource(Wikipedia, 'wikipedia/<article_name>')
        self.add_resource(Coreference, 'coreference/<article_name>')
        self.add_resource(OpenIE, 'openie/<article_name>')
        self.add_resource(Extractions, 'extraction/<file_type>/<article_name>')

        self.add_resource(CacheStatistics, 'cache/statistics')

    # overwrite super to always use /api/ as endpoint
    def add_resource(self, resource, *urls, **kwargs):
        super().add_resource(resource, *['/api/' + u for u in urls], **kwargs)


class CachedResource(Resource):
    def __init__(self):
        super().__init__()
        self.cache = ExtractionCache(DATA_PATH)

    @staticmethod
    def temp_path(article_name):
        return os.path.join(DATA_PATH, hashlib.md5(article_name.encode('UTF-8')).hexdigest())


class CacheStatistics(CachedResource):
    def get(self):
        return self.cache.get_statistics()


class Wikipedia(CachedResource):
    def get(self, article_name):
        try:
            try:
                content = self.cache.get_wiki(article_name)
                notification = 'loaded from cache'
            except (ExtractionCache.CacheOutOfDateException, ExtractionCache.CacheMissedException) as e:
                content = wiki.get_content(article_name)
                self.cache.put_wiki(article_name, content)

                if isinstance(e, ExtractionCache.CacheOutOfDateException):
                    notification = 'Wiki was out of date'
                elif isinstance(e, ExtractionCache.CacheMissedException):
                    notification = 'Wiki was not in cache'
                else:
                    notification = 'something went wrong in wiki cache'

            return {'article': article_name,
                    'content': content,
                    'notification': notification
                    }, req.codes.ok

        except wiki.PageNotFoundException as e:
            return {'error': 'error',
                    'message': e.message
                    }, req.codes.not_found


class Coreference(CachedResource):
    def get(self, article_name):
        try:
            corefs = self.cache.get_coref(article_name)
            notification = 'loaded from cache'
        except (ExtractionCache.CacheOutOfDateException, ExtractionCache.CacheMissedException) as e:
            wiki_content, code = Wikipedia().get(article_name)
            if code != req.codes.ok or 'content' not in wiki_content.keys():
                return wiki_content, code  # error occurred

            wiki_content = wiki_content['content']
            corefs = coref.do_coreference(wiki_content, data_path=os.path.join(DATA_PATH, self.temp_path(article_name)),
                                          max_memory=MAX_MEMORY)
            self.cache.put_coref(article_name, corefs)

            if isinstance(e, ExtractionCache.CacheOutOfDateException):
                notification = 'Coref was out of date'
            elif isinstance(e, ExtractionCache.CacheMissedException):
                notification = 'Coref was not in cache'
            else:
                notification = 'something went wrong in coref cache'

        return {'article': article_name,
                'corefs': corefs,
                'notification': notification
                }, req.codes.ok


class OpenIE(CachedResource):
    def get(self, article_name):
        try:
            openie_output = self.cache.get_openie(article_name)
            notification = 'loaded from cache'
        except (ExtractionCache.CacheOutOfDateException, ExtractionCache.CacheMissedException) as e:
            if isinstance(e, ExtractionCache.CacheOutOfDateException):
                notification = 'OpenIE was out of date'
            elif isinstance(e, ExtractionCache.CacheMissedException):
                notification = 'OpenIE was not in cache'
            else:
                notification = 'something went wrong in OpenIE cache'

            corefs, code = Coreference().get(article_name)
            if code != req.codes.ok or 'corefs' not in corefs.keys():
                return corefs, code  # error occurred

            corefs = corefs['corefs']
            openie_output = openie.do_openie(corefs, data_path=os.path.join(DATA_PATH, self.temp_path(article_name)),
                                             max_memory=MAX_MEMORY)
            self.cache.put_openie(article_name, openie_output)

        return {'article': article_name,
                'openie': openie_output,
                'notification': notification
                }, req.codes.ok


class Extractions(CachedResource):
    ALLOWED_FILE_TYPES = ['e2rdf', 'nt']

    def get(self, file_type, article_name):
        if file_type not in Extractions.ALLOWED_FILE_TYPES:
            return {
                       'error': 'error',
                       'message': 'this file type is not allowed, use: ' + repr(Extractions.ALLOWED_FILE_TYPES)
            }, req.codes.bad_request
        nt, raw = None, None

        try:
            if file_type == 'nt':
                nt = self.cache.get_nt(article_name)
            elif file_type == 'e2rdf':
                raw = self.cache.get_e2rdf(article_name)
            notification = 'loaded from Extraction cache'
        except (ExtractionCache.CacheOutOfDateException, ExtractionCache.CacheMissedException) as e:
            if isinstance(e, ExtractionCache.CacheOutOfDateException):
                notification = 'Extraction was out of date'
            elif isinstance(e, ExtractionCache.CacheMissedException):
                notification = 'Extraction was not in cache'
            else:
                notification = 'something went wrong in Extraction cache'

            openie_output, code = OpenIE().get(article_name)
            if code != req.codes.ok or 'openie' not in openie_output:
                return {'error': 'error',
                        'message': 'extraction can not call openie'
                        }, code

            openie_output = openie_output['openie']

            # generate e2rdf and nt
            raw, nt = e2rdf.do_e2rdf(openie_output, data_path=os.path.join(DATA_PATH, self.temp_path(article_name)))

            self.cache.put_e2rdf(article_name, raw)
            self.cache.put_nt(article_name, nt)

        if file_type == 'e2rdf':
            return {'article': article_name,
                    'e2rdf': raw,
                    'notification': notification
                    }, req.codes.ok

        elif file_type == 'nt':
            return {'article': article_name,
                    'nt': nt,
                    'notification': notification
                    }, req.codes.ok

        else:
            return {'error': 'error',
                    'message': 'this should never be reached'
                    }, req.codes.internal_server_error
