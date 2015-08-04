__author__ = 'bernhard'

from tornado.wsgi import WSGIContainer
from tornado.httpserver import HTTPServer
from tornado.ioloop import IOLoop
from qa_jbt.jbt_app import app

http_server = HTTPServer(WSGIContainer(app))
http_server.listen(8888)
IOLoop.current().start()