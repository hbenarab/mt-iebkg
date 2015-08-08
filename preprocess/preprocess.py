__author__ = 'heni'


import subprocess



# running jbt_server.py

def run_jbt_server():
    with open('../qa-jbt/jbt_server.py') as f:
        code = compile(f.read(), 'jbt_server.py', 'exec')
        exec(code)
    f.close()

def run_command(cmd):
    subprocess.Popen(cmd,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                            stdin=subprocess.PIPE).communicate()

def get_openie(article_name):
    run_jbt_server()
    url='http://localhost:8888/api/openie/'+str(article_name)
    cmd="curl "+url
    run_command(cmd)


get_openie('Sfax')

