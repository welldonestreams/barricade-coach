"""Reuse a compatible local coach, or start one without colliding with old servers."""
import json
from urllib.request import urlopen
import server


def main():
    for port in (8810,8811):
        try:
            with urlopen(f'http://127.0.0.1:{port}/api/health',timeout=1) as response:
                health=json.load(response)
            if health.get('service')=='barricade-coach' and health.get('live_protocol',0)>=6:
                print(f'Updated coach already running: http://127.0.0.1:{port}',flush=True)
                return
        except (OSError,ValueError):pass
    for port in (8810,8811):
        try:httpd=server.LocalServer(('127.0.0.1',port),server.Handler)
        except OSError:continue
        print(f'Updated coach running: http://127.0.0.1:{port}',flush=True)
        httpd.serve_forever()
        return
    raise SystemExit('Both coach ports are occupied by older servers. Close those servers and run this launcher again.')


if __name__=='__main__':main()
