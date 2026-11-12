#!/usr/bin/env python3
import os
from http.server import HTTPServer, SimpleHTTPRequestHandler

class ETagRequestHandler(SimpleHTTPRequestHandler):
    def send_head(self):
        path = self.translate_path(self.path)
        if os.path.isdir(path):
            return super().send_head()

        try:
            f = open(path, 'rb')
        except OSError:
            self.send_error(404, "File not found")
            return None

        fs = os.fstat(f.fileno())
        # Generate a stable ETag based on mtime (nanoseconds) + size to avoid same-second granularity issues
        etag = f'W/"{fs.st_mtime_ns:x}-{fs.st_size:x}"'  # weak ETag is sufficient
        ims = self.headers.get('If-Modified-Since')
        inm = self.headers.get('If-None-Match')

        # If ETag matches, return 304 directly
        if inm and inm == etag:
            f.close()
            self.send_response(304)
            self.send_header("ETag", etag)
            self.send_header("Cache-Control", "public, max-age=0, must-revalidate")
            self.end_headers()
            return None

        # Otherwise send 200 with content
        self.send_response(200)
        ctype = self.guess_type(path)
        self.send_header("Content-type", ctype)
        self.send_header("Content-Length", str(fs.st_size))
        self.send_header("Last-Modified", self.date_time_string(fs.st_mtime))
        self.send_header("ETag", etag)
        # Choose cache policy as needed; use "must-revalidate" here to ensure updates are visible
        self.send_header("Cache-Control", "public, max-age=0, must-revalidate")
        self.end_headers()
        return f

def run(addr="0.0.0.0", port=9999, directory="."):
    handler = ETagRequestHandler
    def factory(*args, **kwargs):
        return handler(*args, directory=directory, **kwargs)
    httpd = HTTPServer((addr, port), factory)
    print(f"Serving on http://{addr}:{port} dir={os.path.abspath(directory)}")
    httpd.serve_forever()

if __name__ == "__main__":
    run()
