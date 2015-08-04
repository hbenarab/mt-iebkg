__author__ = 'bernhard'

import datetime
import requests as req
import sys
import re
import urllib

URL = "/extraction/nt"

PATTERN = r'<http://www.w3.org/2000/01/rdf-schema#label> "(.*)"@en \.$'

REPLACE = [
    r'/'
]

def main(host, article_file):

    host += URL

    not_ok = []
    total_time = datetime.timedelta()
    for article in get_article_titles(article_file):
        print("""[{}]:\t""".format(article), end="")
        try:
            r = req.get(host + __clean_article_name(article))
            total_time += r.elapsed
            if not r.ok or 'error' in r.json():
                not_ok.append(article)
            print("""{}\t{}""".format(r.status_code, str(r.elapsed)))
        except req.ConnectionError:
            print("""ConnectionError""")
            not_ok.append(article)
            continue
        except ConnectionResetError:
            not_ok.append(article)
            error_count += 1
            if error_count > MAX_ERRORS:
                print('There were {} errors, breaking now!')
                break

    print('Queue finished in {}'.format(str(total_time)))
    if len(not_ok) > 0:
        print('Following articles gave errors:')
        print(not_ok)
    else:
        print('No visible error occurred.')


def get_article_titles(article_file):
    with open(article_file, mode='r', encoding='utf-8') as input_file:
        for line in input_file:
            if not line.startswith("#"):
                yield re.findall(PATTERN, line)[0]
            else:
                continue


def __clean_article_name(article):
    for pat in REPLACE:
        article = re.sub(pattern=pat, repl=urllib.parse.quote_plus(r'\1'), string=article)
    return article


if __name__ == '__main__':
    if len(sys.argv) == 3:
        main(host=sys.argv[1], article_file=sys.argv[2])
    else:
        print("Give host and .nt-file (dbpedia-style) as input")
