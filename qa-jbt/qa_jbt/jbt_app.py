__author__ = 'bernhard'

from qa_jbt.web_app.qa_jbt_app import JbtApp
from qa_jbt.web_api.qa_jbt_api import JbtApi

# the app
app = JbtApp()

# the api
api = JbtApi(app)
