__author__ = 'bernhard'

from flask import Flask


class JbtApp(Flask):

    def __init__(self, *args, **kwargs):
        super().__init__(__name__, *args, **kwargs)
