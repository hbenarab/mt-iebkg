__author__ = 'heni'


# running jbt_server.py

def run_jbt_server():
    with open('qa-jbt/jbt_server.py') as f:
        code = compile(f.read(), 'jbt_server.py', 'exec')
        exec(code)

    f.close()

